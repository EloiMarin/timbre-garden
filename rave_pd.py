"""03. Corpus-based analysis and synthesis
"""

import numpy as np
import sklearn.cluster
import sklearn.decomposition
import sklearn.preprocessing

import taichi as ti
from tolvera import Tolvera, run
#from tolvera.osc.update import Updater
# from tolvera.utils import ti_map_range
from functools import reduce
from operator import add
from math import floor
from hilbertcurve.hilbertcurve import HilbertCurve

def main(**kwargs):
    tv = Tolvera(**kwargs)

    # Start with stopped particles
    for i in range(tv.p.n):
        tv.p.field[i].speed = 0.0

    rave_input_dimensions = 16

    p = 16
    n = 2
    hilbert_curve = HilbertCurve(p, n)
    max_hilbert_input = 2**p - 1
    max_hilbert_output = 2**(n * p) - 1

    position_vector = ti.Vector.field(n=2, dtype=ti.i32, shape=(rave_input_dimensions))
    particle_speeds = ti.Vector.field(n=2, dtype=ti.f32, shape=(tv.p.n))
    mean_speed = ti.field(ti.f32, shape=())
    blow_particles = ti.field(ti.f32, shape=())
    friction_coeff = ti.field(ti.f32, shape=())
    friction_coeff[None] = 0.01

    @ti.func
    def ti_map_range(val, in_min, in_max, out_min, out_max):
        return out_min + ((val - in_min) * (out_max - out_min)) / (in_max - in_min)
    
    def RaveOSC():
        @tv.osc.map.send_list(vector=(0.,0.,100.), count=1, length=rave_input_dimensions, send_mode="broadcast")
        def rave_dimensions() -> list[float]:
            return [
                distance / max_hilbert_output for distance in
                hilbert_curve.distances_from_points(position_vector.to_numpy())
            ]

        @tv.osc.map.send_args(arg=(0.,0.,100.), count=1, send_mode="broadcast")
        def rave_speed() -> float:
            return [mean_speed[None]]

        @tv.osc.map.receive_args(arg=(0,1,1), count=1)
        def reset(args):
            if (args != 0):
                tv.randomise()
        @tv.osc.map.receive_args(arg=(0.,0.,100.), count=1)
        def blow(args):
            if (args > 0):
                blow_particles[None] = args
        @tv.osc.map.receive_args(arg=(0.,0.,100.), count=1)
        def friction(args):
            friction_coeff[None] = args
    RaveOSC()

    window_shape = tv.ti.window.get_window_shape()

    @ti.kernel
    def update_mean_speed(p_field: ti.template()):
        a = 0.0
        for i in range(tv.p.n):
            a += ti.math.sqrt(ti.math.pow(p_field[i].vel[0] * p_field[i].speed, 2) + ti.math.pow(p_field[i].vel[1] * p_field[i].speed, 2))
        a /= tv.p.n
        mean_speed[None] = a

    @ti.kernel
    def update_particle_speeds(p_field: ti.template()):
        for i in range(tv.p.n):
            particle_speeds[i] = p_field[i].vel * p_field[i].speed

    @ti.kernel
    def update_position_vector(p_field: ti.template()):
        for i in ti.static(range(rave_input_dimensions)):
            x = ti_map_range(p_field[i].pos[0], 0, window_shape[0], 0, max_hilbert_input)
            y = ti_map_range(p_field[i].pos[1], 0, window_shape[1], 0, max_hilbert_input)
            position_vector[i].xy = (ti.i32(ti.math.round(x)), ti.i32(ti.math.round(y)))
            position_vector[i].xy = ti.math.clamp(position_vector[i], 0, max_hilbert_input) # window_shape might not be updated, so make sure we stay within limits

    @ti.kernel
    def draw_particle_dists(p: ti.template(), s: ti.template(), f: ti.template(), fd: ti.template(), fs: ti.template(), max_radius: ti.f32):
        for i, j in ti.ndrange(tv.p.n,tv.p.n):
            if i==j: continue
            p1 = p[i]
            p2 = p[j]
            if p1.species != p2.species: continue
            sp = s[p1.species]
            alpha = ((p1.vel + p2.vel)/2).norm()
            p1x = ti.cast(p1.pos[0], ti.i32)
            p1y = ti.cast(p1.pos[1], ti.i32)
            p2x = ti.cast(p2.pos[0], ti.i32)
            p2y = ti.cast(p2.pos[1], ti.i32)
            tv.px.circle(p1x, p1y, 2 * alpha, sp.rgba)
            d = fd[i,j].dist
            r = fs[p1.species,p2.species].radius
            if d > r * max_radius: continue
            # hack, should draw two lines when wrapping occurs
            if ti.abs(p1x - p2x) > tv.x-fs[i,j].radius: continue
            if ti.abs(p1y - p2y) > tv.y-fs[i,j].radius: continue
            tv.px.line(p1x, p1y, p2x, p2y, sp.rgba)

    @ti.kernel
    def friction(particles: ti.template()):
        for i in range(tv.p.n):
            if particles.field[i].active == 0:
                continue
            if (blow_particles[None] > 0):
                particles.field[i].speed += ti.randn() * blow_particles[None]
            else:
                speed = particles.field[i].speed * (1 - friction_coeff[None])
                if (speed < 0.001):
                    speed = 0.0
                particles.field[i].speed = speed
        blow_particles[None] = 0

    @tv.render
    def _():
        tv.px.diffuse(0.99)

        tv.v.flock(tv.p)

        friction(tv.p)
        
        # Attract particles to a position with a mass and radius
        #tv.v.attract(tv.p, [tv.x/2, tv.y/2], 10.0, tv.y)

        # Attract particle species to a position with a mass and radius
        #tv.v.attract_species(tv.p, [tv.x/2, tv.y/2], 10.0, tv.y, 1)

        # Repel particles from a position with a mass and radius
        # tv.v.repel(tv.p, [tv.x/2, tv.y/2], 10.0, tv.y)

        # Repel particle species from a position with a mass and radius
        #tv.v.repel_species(tv.p, [tv.x/2, tv.y/2], 10.0, tv.y, 1)

        # Gravitate particles to a position with force G and radius
        # tv.v.gravitate(tv.p, 10.0, 100.0)

        # Gravitate particle species to a position with force G and radius
        # tv.v.gravitate_species(tv.p, 10.0, 100.0, 0)

        # Add noise to the particles with a weight (scalar)
        # tv.v.noise(tv.p, 1.0)

        # ti_funcs()

        tv.px.particles(tv.p, tv.s.species())
        #draw_particle_dists(tv.p.field, tv.s.species(), tv.s.flock_p(), tv.s.flock_dist(), tv.s.flock_s(), tv.v.flock.CONSTS.MAX_RADIUS)

        update_position_vector(tv.p.field)
        update_particle_speeds(tv.p.field)
        update_mean_speed(tv.p.field)

        return tv.px

if __name__ == '__main__':
    run(main)

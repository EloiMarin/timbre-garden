def RaveOSC():
    collision_field = ti.field(ti.i32, shape=tv.s.grains.shape[0])
    collision_field.fill(0)

    @ti.kernel
    def detect_collisions(radius: ti.f32):
        for gi, pi in ti.ndrange(tv.s.grains.shape[0], tv.p.field.shape[0]):
            collision_field[gi] = 0
            g = ti.Vector(grain_pos_f32(gi))
            p = tv.p.field[pi]
            if p.active == 0:
                continue
            dist = p.pos - g
            if dist.norm() < radius:
                collision_field[gi] = 1

    @tv.osc.map.send_list(vector=(0.,0.,100.), count=1, length=16, send_mode="broadcast")
    def rave_dimensions() -> list[float]:
        detect_collisions(10)
        np_collision_field = collision_field.to_numpy()

        message = []
        dimension_size = floor(len(np_collision_field) / 16)
        for dim in range(0, 16):
            base = dim * dimension_size
            message.append(reduce(add, np_collision_field[base:base + dimension_size]))

        return map(float, message)

    @tv.osc.map.receive_args(arg=(0,1,1), count=1)
    def tolvera_reset(args):
        if (args != 0):
            tv.randomise()

from destlib import *
from Ant import Ant


if __name__ == "__main__":
    cars1 = DES() \
        .set_sample_size(12) \
        .set_time_between_distro(uniform, a=1, b=7) \
        .set_service_time_distro(uniform, a=1, b=4) \
        .set_seed(7122004) \
        .set_system_name("TollBooth") \
        .set_entity_name("Car") \
        .set_time_unit(TimeUnit.Min)

    for i, car in enumerate(run_simulations(cars1, 3)):
        car.run()
        car.save_to("csv")
        car.plot(v_lines=True)
        print(car.df["start_time"].mean())
        print(f"{i} is done")

    dst = DST() \
        .set_sim_class(Ant) \
        .set_behaviors(
            [
                Ant.move_up,
                Ant.move_down,
                Ant.move_left,
                Ant.move_right
            ]
    ).set_behaviors_calls(10)

    ant = dst.run(state_history=True)
    for state in ant:
        print(state)

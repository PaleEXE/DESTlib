from matplotlib.pylab import poisson
from destlib import DES, run_simulations, uniform, TimeUnit
from destlib import DES, uniform, TimeUnit


if __name__ == "__main__":
    """cars1 = (
        DES()
        .set_sample_size(30)
        .set_time_between_distro(uniform, a=1, b=7)
        .set_service_time_distro(uniform, a=1, b=7)
        .set_seed(7122004)
        .set_system_name("TollBooth")
        .set_entity_name("Car")
        .set_time_unit(TimeUnit.Min)
    )

    for i, car in enumerate(run_simulations(cars1, 3)):  # noqa: F405
        car.set_num_servers(i + 1)  # type: ignore
        car.run()
        car.plot(v_lines=True)  # type: ignore"""

    patiants = (
    patients = (
        DES()
        .set_sample_size(100)
        .set_time_between_distro(poisson, lam=7.0)
        .set_service_time_distro(uniform, a=1, b=30)
        .set_num_servers(3)
        .set_num_servers(30)
        .set_levels(
            levels=["Critical", "Severe", "Moderate"],
            levels_prob=[0.1, 0.3, 0.6],
        )
        .set_seed(7122004)
        .set_system_name("Hospital")
        .set_entity_name("Patient")
        .set_time_unit(TimeUnit.Min)
    )

    patiants.run()

    print(patiants.df.groupby("level")["wait_time"].mean())
    print(patients.df.groupby("level")["wait_time"].mean())

from matplotlib.pylab import poisson
from destlib import DES, uniform, TimeUnit


if __name__ == "__main__":
    patients = (
        DES()
        .set_sample_size(30)
        .set_time_between_distro(poisson, lam=8.0)
        .set_service_time_distro(uniform, a=1, b=30)
        .set_num_servers(3)
        .set_levels(
            levels=["Critical", "Severe", "Moderate"],
            levels_prob=[0.2, 0.4, 0.4],
        )
        .set_seed(7122004)
        .set_system_name("Hospital")
        .set_entity_name("Patient")
        .set_time_unit(TimeUnit.Min)
    )

    patients.run()
    patients.plot()

    print(patients.df.groupby("level")["wait_time"].mean())
    
    patients.save_to("csv", True, True)

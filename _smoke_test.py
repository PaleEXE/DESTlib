import sys
sys.path.insert(0, ".")

try:
    import destlib
    print("Import OK")
    
    from destlib import DES, TimeUnit, uniform
    print("Named imports OK")
    
    sim = DES()
    sim.set_sample_size(100).set_seed(42)
    sim.set_time_between_distro(uniform, a=1, b=5)
    sim.set_service_time_distro(uniform, a=1, b=3)
    sim.set_num_servers(2)
    sim.run()
    print(f"Run OK, DataFrame shape: {sim.df.shape}")
    
    stats = sim.compute_statistics()
    print(f"Statistics OK: {stats}")
    
    print(f"len(sim) = {len(sim)}")
    
    rows = list(sim)
    print(f"Iteration OK, got {len(rows)} rows")
    
    print("\nAll checks passed!")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

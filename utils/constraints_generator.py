from utils.logger_config import get_logger
logger = get_logger("CONST GEN")

def generate_constraints(df, num_vnfs):
    logger.info("Generating Constraints")
    #Define Constraints
    CHAIN_LEN = num_vnfs # TODO : Different Slices --> Different Chains

    cpu_high = df['cpu_cycles'].quantile(0.85)
    mem_high = df['memory_mb'].quantile(0.85)
    lat_mean = df['latency_ms'].mean()

    CPU_BUDGET   = int(CHAIN_LEN * cpu_high * 0.85)
    MEM_BUDGET   = int(CHAIN_LEN * mem_high * 0.80)
    LATENCY_SLA  = CHAIN_LEN * lat_mean * 1.10
    GPU_BUDGET   = 2

    constraints = {
        "CPU_BUDGET": CPU_BUDGET,
        "MEM_BUDGET": MEM_BUDGET,
        "GPU_BUDGET": GPU_BUDGET,
        "LATENCY_SLA": LATENCY_SLA,
    }

    logger.info(f"CPU_BUDGET: {CPU_BUDGET}")
    logger.info(f"MEM_BUDGET: {MEM_BUDGET}")
    logger.info(f"GPU_BUDGET: {GPU_BUDGET}")
    logger.info(f"LATENCY_SLA: {LATENCY_SLA}")

    logger.info("Constraints Generated")
    return constraints
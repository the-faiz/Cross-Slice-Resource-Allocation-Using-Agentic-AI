from data.vnf_dataset import load_vnf_data
from utils.constraints_generator import generate_constraints

from env.vnf_environment import VNFEnvironment
from agents.policy_gradient_agent import PolicyGradientAgent
from rewarders.composite_qos_rewarder import CompositeQoSRewarder

from utils.logger_config import setup_logger
setup_logger("logs/run.log") 

def main():
    print("Loading Data...")
    df, vnf_list, vnf_to_models = load_vnf_data("vnf_model_catalog.csv")
    NUM_VNFS = len(vnf_list)
    # return

    print("Loading Constraints...")
    constraints = generate_constraints(df, NUM_VNFS)

    print("Loading VNF Priority...")
    vnf_priority = {f"VNF_{i:02d}": 1.0 + 0.05*(i-1) for i in range(1, 21)} # TODO : Replace it with loading from a config

    print("Intializing the Rewarder Algorithm...")
    rewarder = CompositeQoSRewarder()

    print("Intializing the VNF Environment...")
    env = VNFEnvironment(vnf_list, vnf_to_models, vnf_priority, constraints, rewarder)

    print("Intializing the Agent...")
    agent = PolicyGradientAgent(env)

    print("Training the Agent...")
    agent.train()

    print("Evaluating the Agent...")
    agent.evaluate_agent()
    agent.print_selection_example()


if __name__ == "__main__":
    main()

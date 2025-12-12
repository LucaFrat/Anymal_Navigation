import torch

# REPLACE WITH YOUR EXACT PATH
PATH = "source/Anymal_Navigation/Anymal_Navigation/isaaclab_tasks/manager_based/navigation/config/anymal_c/agents/policy.pt"

try:
    model = torch.jit.load(PATH)
    print("✅ Model loaded!")

    # Inspect the input of the forward method
    graph = model.inlined_graph
    inputs = list(graph.inputs())
    print(f"Number of inputs: {len(inputs)}")

    # Usually input 0 is 'self', input 1 is 'obs'
    if len(inputs) > 1:
        obs_input = inputs[1]
        print(f"Observation Input Details: {obs_input}")
        print(f"Type: {obs_input.type()}")

except Exception as e:
    print(f"❌ Error: {e}")
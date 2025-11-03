# This is a sample Python script.
import torch
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from src.neural_ssm import DeepSSM

seed = 9
torch.manual_seed(seed)



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    x = DeepSSM(d_input=1, d_output=1, d_state=2, d_model=2, param='l2ru', ff='LMLP', n_layers=3)
    input = torch.randn(3, 3, 1)
    output = torch.zeros_like(input)
    s= torch.zeros(3, 2)

    for t in range(input.shape[1]):
        output[:,t:t+1,:], s = x(input[:,t:t+1,:], state = s, mode='scan')
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

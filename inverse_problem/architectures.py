import torch
import einops

class Cantilever(torch.nn.Module):
    def __init__(self, number_of_convs1=32, number_of_convs2=64, hidden_size=128, dropout_ratio=0.5):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, number_of_convs1, kernel_size=3, padding='same')
        self.leakyrelu1 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.maxpool1 = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(number_of_convs1, number_of_convs2, kernel_size=3, padding='same')
        self.leakyrelu2 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.maxpool2 = torch.nn.MaxPool1d(2)

        self.linear1 = torch.nn.Linear(number_of_convs2 * 75, hidden_size)
        self.leakyrelu3 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.dropout = torch.nn.Dropout1d(p=dropout_ratio)
        self.linear2 = torch.nn.Linear(hidden_size, 5)

    def forward(self, input_tsr):  # input_tsr.shape = (B, 1, 301)
        act1 = self.conv1(input_tsr)  # (B, C1, 301)
        act2 = self.leakyrelu1(act1)  # (B, C1, 301)
        act3 = self.maxpool1(act2)  # (B, C1, 150)
        act4 = self.conv2(act3)  # (B, C2, 150)
        act5 = self.leakyrelu2(act4)  # (B, C2, 150)
        act6 = self.maxpool2(act5)  # (B, C2, 75)
        act7 = einops.rearrange(act6, 'B C D -> B (C D)')  # (B, C2 * 75)
        act8 = self.linear1(act7)  # (B, H)
        act9 = self.leakyrelu3(act8)  # (B, H)
        act10 = self.dropout(act9)  # (B, H)
        act11 = self.linear2(act10)  # (B, 5)
        return act11

class Balance(torch.nn.Module):
    def __init__(self, number_of_convs1=32, number_of_convs2=64, hidden_size=128, dropout_ratio=0.5):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, number_of_convs1, kernel_size=3, padding='same')
        self.tanh1 = torch.nn.Tanh()
        self.maxpool1 = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(number_of_convs1, number_of_convs2, kernel_size=3, padding='same')
        self.tanh2 = torch.nn.Tanh()
        self.maxpool2 = torch.nn.MaxPool1d(2)

        self.linear1 = torch.nn.Linear(number_of_convs2 * 75, hidden_size)
        self.tanh3 = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout1d(p=dropout_ratio)
        self.linear2 = torch.nn.Linear(hidden_size, 5)

    def forward(self, input_tsr):  # input_tsr.shape = (B, 1, 301)
        act1 = self.conv1(input_tsr)  # (B, C1, 301)
        act2 = self.tanh1(act1)  # (B, C1, 301)
        act3 = self.maxpool1(act2)  # (B, C1, 150)
        act4 = self.conv2(act3)  # (B, C2, 150)
        act5 = self.tanh2(act4)  # (B, C2, 150)
        act6 = self.maxpool2(act5)  # (B, C2, 75)
        act7 = einops.rearrange(act6, 'B C D -> B (C D)')  # (B, C2 * 75)
        act8 = self.linear1(act7)  # (B, H)
        act9 = self.tanh3(act8)  # (B, H)
        act10 = self.dropout(act9)  # (B, H)
        act11 = self.linear2(act10)  # (B, 5)
        return act11

class Coil(torch.nn.Module):
    def __init__(self, number_of_convs1=32, number_of_convs2=64, hidden_size=128, dropout_ratio=0.5):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, number_of_convs1, kernel_size=3, padding='same')
        self.maxpool1 = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(number_of_convs1, number_of_convs2, kernel_size=3, padding='same')
        self.maxpool2 = torch.nn.MaxPool1d(2)

        self.linear1 = torch.nn.Linear(number_of_convs2 * 75, hidden_size)
        self.dropout = torch.nn.Dropout1d(p=dropout_ratio)
        self.linear2 = torch.nn.Linear(hidden_size, 5)

    def forward(self, input_tsr):  # input_tsr.shape = (B, 1, 301)
        act1 = self.conv1(input_tsr)  # (B, C1, 301)
        act2 = torch.sin(act1)  # (B, C1, 301)
        act3 = self.maxpool1(act2)  # (B, C1, 150)
        act4 = self.conv2(act3)  # (B, C2, 150)
        act5 = torch.sin(act4)  # (B, C2, 150)
        act6 = self.maxpool2(act5)  # (B, C2, 75)
        act7 = einops.rearrange(act6, 'B C D -> B (C D)')  # (B, C2 * 75)
        act8 = self.linear1(act7)  # (B, H)
        act9 = torch.sin(act8)  # (B, H)
        act10 = self.dropout(act9)  # (B, H)
        act11 = self.linear2(act10)  # (B, 5)
        return act11

class Leaf(torch.nn.Module):
    def __init__(self, number_of_convs1=32, number_of_convs2=64, hidden_size=128, dropout_ratio=0.5):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, number_of_convs1, kernel_size=3, padding='same')

        self.maxpool1 = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(number_of_convs1, number_of_convs2, kernel_size=3, padding='same')
        self.tanh2 = torch.nn.Tanh()
        self.maxpool2 = torch.nn.MaxPool1d(2)

        self.linear1 = torch.nn.Linear(number_of_convs2 * 75, hidden_size)
        self.tanh3 = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout1d(p=dropout_ratio)
        self.linear2 = torch.nn.Linear(hidden_size, 5)

    def forward(self, input_tsr):  # input_tsr.shape = (B, 1, 301)
        act1 = self.conv1(input_tsr)  # (B, C1, 301)
        act2 = torch.sin(act1)  # (B, C1, 301)
        act3 = self.maxpool1(act2)  # (B, C1, 150)
        act4 = self.conv2(act3)  # (B, C2, 150)
        act5 = self.tanh2(act4)  # (B, C2, 150)
        act6 = self.maxpool2(act5)  # (B, C2, 75)
        act7 = einops.rearrange(act6, 'B C D -> B (C D)')  # (B, C2 * 75)
        act8 = self.linear1(act7)  # (B, H)
        act9 = self.tanh3(act8)  # (B, H)
        act10 = self.dropout(act9)  # (B, H)
        act11 = self.linear2(act10)  # (B, 5)
        return act11

class Volute(torch.nn.Module):
    def __init__(self, number_of_convs1=32, number_of_convs2=64, hidden_size=128, dropout_ratio=0.5):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, number_of_convs1, kernel_size=7, padding='same')
        self.leakyrelu1 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.maxpool1 = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(number_of_convs1, number_of_convs2, kernel_size=7, padding='same')
        self.leakyrelu2 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.maxpool2 = torch.nn.MaxPool1d(2)

        self.linear1 = torch.nn.Linear(number_of_convs2 * 75, hidden_size)
        self.leakyrelu3 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.dropout = torch.nn.Dropout1d(p=dropout_ratio)
        self.linear2 = torch.nn.Linear(hidden_size, 5)

    def forward(self, input_tsr):  # input_tsr.shape = (B, 1, 301)
        act1 = self.conv1(input_tsr)  # (B, C1, 301)
        act2 = self.leakyrelu1(act1)  # (B, C1, 301)
        act3 = self.maxpool1(act2)  # (B, C1, 150)
        act4 = self.conv2(act3)  # (B, C2, 150)
        act5 = self.leakyrelu2(act4)  # (B, C2, 150)
        act6 = self.maxpool2(act5)  # (B, C2, 75)
        act7 = einops.rearrange(act6, 'B C D -> B (C D)')  # (B, C2 * 75)
        act8 = self.linear1(act7)  # (B, H)
        act9 = self.leakyrelu3(act8)  # (B, H)
        act10 = self.dropout(act9)  # (B, H)
        act11 = self.linear2(act10)  # (B, 5)
        return act11

class Vspring(torch.nn.Module):
    def __init__(self, number_of_convs1=32, number_of_convs2=64,
                 number_of_convs3=128, number_of_convs4=256,
                 hidden_size=128, dropout_ratio=0.5):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, number_of_convs1, kernel_size=7, padding='same')
        self.leakyrelu1 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.maxpool1 = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(number_of_convs1, number_of_convs2, kernel_size=7, padding='same')
        self.leakyrelu2 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.maxpool2 = torch.nn.MaxPool1d(2)
        self.conv3 = torch.nn.Conv1d(number_of_convs2, number_of_convs3, kernel_size=7, padding='same')
        self.leakyrelu3 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.maxpool3 = torch.nn.MaxPool1d(2)
        self.conv4 = torch.nn.Conv1d(number_of_convs3, number_of_convs4, kernel_size=7, padding='same')
        self.leakyrelu4 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.maxpool4 = torch.nn.MaxPool1d(2)

        self.linear1 = torch.nn.Linear(number_of_convs4 * 18, hidden_size)
        self.leakyrelu5 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.dropout = torch.nn.Dropout1d(p=dropout_ratio)
        self.linear2 = torch.nn.Linear(hidden_size, 5)

    def forward(self, input_tsr):  # input_tsr.shape = (B, 1, 301)
        act1 = self.conv1(input_tsr)  # (B, C1, 301)
        act2 = self.leakyrelu1(act1)  # (B, C1, 301)
        act3 = self.maxpool1(act2)  # (B, C1, 150)
        act4 = self.conv2(act3)  # (B, C2, 150)
        act5 = self.leakyrelu2(act4)  # (B, C2, 150)
        act6 = self.maxpool2(act5)  # (B, C2, 75)
        act7 = self.conv3(act6)  # (B, C3, 75)
        act8 = self.leakyrelu3(act7)  # (B, C3, 75)
        act9 = self.maxpool3(act8)  # (B, C3, 37)
        act10 = self.conv4(act9)  # (B, C4, 37)
        act11 = self.leakyrelu4(act10)  # (B, C4, 37)
        act12 = self.maxpool4(act11)  # (B, C4, 18)

        act13 = einops.rearrange(act12, 'B C D -> B (C D)')  # (B, C4 * 18)
        act14 = self.linear1(act13)  # (B, H)
        act15 = self.leakyrelu5(act14)  # (B, H)
        act16 = self.dropout(act15)  # (B, H)
        act17 = self.linear2(act16)  # (B, 5)
        return act17
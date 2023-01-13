import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        op = self.output(x)

        return op

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        os.makedirs(model_folder_path, exist_ok=True)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, finish):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            # only one input, make the batch size as 1
            # (1, x)
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)

            finish = (finish, )

        # BELLMAN EQUATION
        # Q = pred(state_0)
        # Q_new = reward + gamma * max(Q(state_1))

        # predicted Q values with the current states
        pred = self.model(state)
        # find the Q_new using new state reward and gamma -> only do this if not done and update the best action only
        target = pred.clone()
        # going through the batch
        for idx in range(len(finish)):
            Q_new = reward[idx]
            if not finish[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # only update the action idx which is taken i.e. straight, left, right
            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)  # Q_new and Q
        # back-prop to calculate gradients
        loss.backward()
        # update the weights
        self.optimizer.step()


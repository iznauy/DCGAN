import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils as utils
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as functional

# cifar-10 image size: 3 * 32 * 32

batch_size = 64
epoch = 20
alpha = 0.2
beta1 = 0.6
beta2 = 0.999
lr = 0.0005
hidden_size = 100

print_every = 10

class Generator(nn.Module):

    initial_size = 512

    def __init__(self, out_dim, alpha): # out_dim = 3

        super(Generator, self).__init__()
        self.out_dim = out_dim
        self.alpha = alpha

        # input: 100 numbers
        # output: 3 * 3 * 512 = 4608
        self.fc1 = nn.Linear(100, 3 * 3 * Generator.initial_size, bias=True)

        # input: 512 * 3 * 3
        self.bn1 = nn.BatchNorm2d(Generator.initial_size)

        # input: 512 * 3 * 3
        # output: 128 * 8 * 8
        self.deconv2 = nn.ConvTranspose2d(Generator.initial_size, Generator.initial_size / 4,
                                         kernel_size=4, stride=2)

        # input: 128 * 8 * 8
        self.bn2 = nn.BatchNorm2d(Generator.initial_size // 4)

        # input: 128 * 8 * 8
        # output: 32 * 18 * 18
        self.deconv3 = nn.ConvTranspose2d(Generator.initial_size // 4, Generator.initial_size // 16,
                                          kernel_size=4, stride=2)

        # input: 32 * 18 * 18
        self.bn3 = nn.BatchNorm2d(Generator.initial_size // 16)

        # input: 32 * 18 * 18
        # output: 3 * 32 * 32
        self.deconv4 = nn.ConvTranspose2d(Generator.initial_size // 16, self.out_dim,
                                          kernel_size=4, stride=2, padding=3)


    def forward(self, input):

        out_1 = self.fc1(input)
        flatted_out_1 = out_1.view(-1, Generator.initial_size, 3, 3)
        normed_out_1 =functional.leaky_relu(self.bn1(flatted_out_1), negative_slope=self.alpha)

        out_2 = functional.leaky_relu(self.bn2(self.deconv2(normed_out_1)), negative_slope=self.alpha)

        out_3 = functional.leaky_relu(self.bn3(self.deconv3(out_2)), negative_slope=self.alpha)

        out_4 = self.deconv4(out_3)

        out = functional.tanh(out_4)

        return out




class Discriminator(nn.Module):

    def __init__(self, in_dim, alpha):

        super(Discriminator, self).__init__()
        self.alpha = alpha
        self.in_dim = in_dim

        # input: 3 * 32 * 32
        # output: 6 * 28 * 28
        self.conv1 = nn.Conv2d(in_dim, 2 * in_dim, kernel_size=5, padding=0)
        self.bn1 = nn.BatchNorm2d(2 * in_dim)

        # input: 6 * 14 * 14
        # output: 12 * 10 * 10
        self.conv2 = nn.Conv2d(in_dim * 2, in_dim * 4, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm2d(4 * in_dim)

        # input: 12 * 5 * 5 = 300
        # output: 10
        self.fc3 = nn.Linear(4 * in_dim * 5 * 5, 10, bias=True)

        # input: 10
        # output: 1
        self.fc4 = nn.Linear(10, 1, bias=True)

        # init parameters
        # To be implements
        # but I don't know how to init it



    def forward(self, input):

        out_1 = functional.max_pool2d(functional.leaky_relu(self.bn1(self.conv1(input)),
                                                            negative_slope=self.alpha), kernel_size=2)

        out_2 = functional.max_pool2d(functional.leaky_relu(self.bn2(self.conv2(out_1)),
                                                            negative_slope=self.alpha), kernel_size=2)
        flatted_out_2 = out_2.view(-1, 4 * self.in_dim * 5 * 5)

        out_3 = functional.leaky_relu(self.fc3(flatted_out_2), negative_slope=self.alpha)

        out_4 = self.fc4(out_3)

        out = functional.sigmoid(out_4)

        return out



def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_loader = utils.data.DataLoader(
        torchvision.datasets.CIFAR10('./Data', train=True, download=False, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    G = Generator(3, alpha)
    D = Discriminator(3, alpha)

    criterion = nn.BCELoss()
    G_optim = torch.optim.Adam(G.parameters(), lr=lr, betas=[beta1, beta2])
    D_optim = torch.optim.Adam(D.parameters(), lr=lr, betas=[beta1, beta2])

    time = 0

    for i in range(epoch):
        for x_, _ in train_loader:

            time += 1

            # train Discriminator
            current_size = x_.size()[0]

            y_real = Variable(torch.ones(current_size))
            y_fake = Variable(torch.zeros(current_size))
            x_ = Variable(x_)

            D_result_real = D(x_)
            D_real_loss = criterion(D_result_real, y_real)

            z1_ = Variable(torch.Tensor(current_size, hidden_size).uniform_(-1, 1))
            x_fake = G(z1_)
            D_result_fake = D(x_fake)
            D_fake_loss = criterion(D_result_fake, y_fake)

            D_loss = D_fake_loss + D_fake_loss

            D.zero_grad()
            D_loss.backward()
            D_optim.step()

            # train Generator
            z2_ = Variable(torch.Tensor(current_size, hidden_size).uniform_(-1, 1))
            y_ = Variable(torch.ones(current_size))
            G_result = G(z2_)
            D_G_result = D(G_result)
            G_loss = criterion(D_G_result, y_)

            G.zero_grad()
            G_loss.backward()
            G_optim.step()

            if time % print_every == 0:
                time = 0
                print "Epoch: ", (i + 1)
                print "Real_loss: ", D_real_loss
                print "Fake_loss: ", G_loss


if __name__ == "__main__":
    main()

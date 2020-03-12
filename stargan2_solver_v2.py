import torchvision
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from model import Generator, Mapping, StyleEncoder, init_weights
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime

writer = SummaryWriter('/data/datasets/starganv2/runs/')


class Solver(object):
    """
    Solver for training and testing StarGAN.
    """

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.num_domains = 2
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        # self.lambda_sty = config.lambda_rec
        self.lambda_sty = 1
        self.lambda_ds = 1
        self.lambda_cyc = 1

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.e_lr = config.e_lr
        self.f_lr = config.f_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs
        self.reg_param = 1

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()

        # if self.use_tensorboard:
        #     self.build_tensorboard()

    def ones_target(self, size, device="cuda"):
        """Tensor containing ones, with shape = size"""
        # data = Variable(torch.ones(size, 1))
        if device == "cuda":
            data = torch.ones((size, 1)).to(device)
        else:
            data = torch.ones((size, 1))
        return data

    def zeros_target(self, size, device="cuda"):
        """
        Tensor containing zeros, with shape = size
        """
        # data = Variable(torch.zeros(size, 1))
        if device == "cuda":
            data = torch.zeros((size, 1)).to(device)
        else:
            data = torch.zeros((size, 1))
        return data

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            # self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.G = Generator(self.g_conv_dim)
            # self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
            self.D = Discriminator(repeat_num=5, channel_multiplier=32, dimension=1)

            self.F = Mapping(image_size=128, repeat_num=6)

            self.E = StyleEncoder(repeat_num=5, channel_multiplier=16, dimension=64)

            # # initialize the weights  of all modules using he init and set all biases to 0
            # self.G.apply(init_weights)
            # self.D.apply(init_weights)
            # self.E.apply(init_weights)
            # self.F.apply(init_weights)

        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim + self.c2_dim + 2, self.g_repeat_num)  # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim + self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])

        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        # self.d_optimizer = torch.optim.SGD(self.D.parameters(), lr=0.0001, momentum=0.9)

        self.e_optimizer = torch.optim.Adam(self.E.parameters(), self.e_lr, [self.beta1, self.beta2])

        self.f_optimizer = torch.optim.Adam(self.F.parameters(), self.f_lr, [self.beta1, self.beta2])

        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        # self.ce_loss = nn.CrossEntropyLoss()

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.E, 'E')
        self.print_network(self.F, 'F')

        self.G.to(self.device)
        self.D.to(self.device)
        self.E.to(self.device)
        self.F.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def noise(self, size, dimension, device="cuda"):
        """
        Generates a 1-d vector of gaussian sampled random values
        """
        # n = Variable(torch.randn(size, 100))
        # n = torch.randn((size, 100), requires_grad=True).to(device)
        if device == "cuda":
            n = torch.randn((size, dimension), requires_grad=True)
        else:
            n = torch.randn((size, dimension), requires_grad=True).to(device)
        return n

    def compute_grad2(self, d_out, x_in):
        """
        https://github.com/LMescheder/GAN_stability/blob/master/gan_training/train.py
        :param d_out: discriminator output
        :param x_in: x_real or x_fake
        :return:
        """
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg

    def train_discriminator(self, x_real, label_org, s_tilde_trg, label_trg):
        """
        train discriminator
        @:param is_d is d or g
        :return:
        """
        self.d_optimizer.zero_grad()

        x_real.requires_grad = True

        # compute adversarial loss on real images
        # with torch.no_grad():
        out_real = torch.gather(self.D(x_real, num_domains=2), 1, label_org.long())
        loss_real = self.bce_loss(out_real, self.ones_target(self.batch_size))
        # loss_real.backward()

        # R1 regularization only on real data
        # out_real.requires_grad = True
        # loss_real.backward(retain_graph=True)

        reg = self.reg_param * self.compute_grad2(out_real, x_real).mean()
        # reg.backward()

        # target style code s_tilde

        # Compute adversarial loss on fake images.
        # x_fake = self.G(x_real, s_tilde[0])
        x_fake = self.G(x_real, s_tilde_trg)
        x_fake.detach()
        d = self.D(x_fake)

        out_fake = torch.gather(d, 1, label_trg.long())
        # d_loss_fake = torch.mean(out_fake)
        loss_fake = self.bce_loss(out_fake, self.zeros_target(self.batch_size))
        # loss_fake.backward()

        # self.d_optimizer.step()

        # Backward and optimize.
        # d_loss = loss_real + loss_fake
        d_loss = loss_real + loss_fake + reg
        d_loss.backward()
        self.d_optimizer.step()

        # l1_norm = torch.norm(self.D.weight, p=1)
        # d_loss += l1_norm

        return d_loss, loss_real, loss_fake

    def train_generator(self,  x_real, label_org, label_trg):
        """
        train generator
        :param x_real:
        :param label_org:
        :param label_trg:
        :return:
        """
        # clear cached gradients for optimizer
        self.g_optimizer.zero_grad()
        self.e_optimizer.zero_grad()
        self.f_optimizer.zero_grad()

        # style reconstruction
        g_s_tilde_trg = self.generate_style_code(label_trg)
        # g_x_fake = self.G(x_real, g_s_tilde_trg)
        # s_hat = self.E(d_x_fake)

        # s_hat: estimated style code of source image
        # loss style reconstruction:style reconstruction
        s_hat_sty = self.E(self.G(x_real, g_s_tilde_trg), num_domains=self.num_domains)
        # s_hat_trg = torch.index_select(torch.stack(s_hat_sty, 1), 1, label_trg.squeeze().long())[:, 0, :]
        s_hat_trg = torch.squeeze(torch.stack([torch.index_select(x, 0, i) for x, i in
                                                 zip(torch.chunk(torch.stack(s_hat_sty, 1), chunks=self.num_domains, dim=1),
                                                     label_trg.squeeze().long())]))

        g_loss_sty = self.l1_loss(g_s_tilde_trg, s_hat_trg)

        # loss cycle: preserving source characteristics
        s_hat_cyc = self.E(x_real, num_domains=self.num_domains)
        # s_hat_org = torch.index_select(torch.stack(s_hat_cyc, 1), 1, label_org.squeeze().long())[:, 0, :]
        s_hat_org = torch.squeeze(torch.stack([torch.index_select(x, 0, i) for x, i in
                                                 zip(torch.chunk(torch.stack(s_hat_cyc, 1), chunks=self.num_domains, dim=1),
                                                     label_org.squeeze().long())]))

        x_fake_cyc = self.G(self.G(x_real, g_s_tilde_trg), s_hat_org)
        g_loss_cyc = self.l1_loss(x_real, x_fake_cyc)

        # loss style diversification:style diversification
        # z1 = self.noise(size=self.batch_size, dimension=16)
        # s1_tilde = self.F(z1, num_domains=2)
        # s1_tilde_trg = torch.index_select(torch.stack(s1_tilde, 1), 1, label_trg.squeeze().long())[:, 0, :]
        s1_tilde_trg = self.generate_style_code(label_trg)
        # z2 = self.noise(size=self.batch_size, dimension=16)
        # s2_tilde = self.F(z2, num_domains=2)
        # s2_tilde_trg = torch.index_select(torch.stack(s2_tilde, 1), 1, label_trg.squeeze().long())[:, 0, :]
        s2_tilde_trg = self.generate_style_code(label_trg)
        g_loss_ds = self.l1_loss(self.G(x_real, s1_tilde_trg), self.G(x_real, s2_tilde_trg))
        #
        # out_real = torch.gather(self.D(x_real, num_domains=2), 1, label_org.long())
        # loss_real = self.bce_loss(self.ones_target(self.batch_size), out_real)
        # target style code s_tilde

        # Compute loss with fake images.
        # x_fake = self.G(x_real, s_tilde[0])

        # compute adversarial loss on fake images
        x_fake = self.G(x_real, g_s_tilde_trg)
        d = self.D(x_fake)
        out_fake = torch.gather(d, 1, label_trg.long())
        # d_loss_fake = torch.mean(out_fake)
        g_adv_loss = self.bce_loss(out_fake, self.ones_target(self.batch_size))

        g_loss = g_adv_loss + self.lambda_sty * g_loss_sty + self.lambda_cyc * g_loss_cyc - self.lambda_ds * g_loss_ds
        # g_loss = g_adv_loss

        g_loss.backward()

        self.g_optimizer.step()
        self.e_optimizer.step()
        self.f_optimizer.step()

        return g_loss, g_adv_loss, g_loss_sty, g_loss_cyc, g_loss_ds
        # return g_loss

    def compute_adversarial_loss(self, is_d, x_real, label_org, s_tilde_trg, label_trg):
        """
        compute non-saturating adversarial loss
        @:param is_d is d or g
        :return:
        """
        out_real = torch.gather(self.D(x_real, num_domains=2), 1, label_org.long())
        loss_real = self.bce_loss(self.ones_target(self.batch_size), out_real)
        # target style code s_tilde

        # Compute loss with fake images.
        # x_fake = self.G(x_real, s_tilde[0])
        x_fake = self.G(x_real, s_tilde_trg)
        if is_d:
            # d = self.D(x_fake).detach()
            x_fake.detach()
        d = self.D(x_fake)

        out_fake = torch.gather(d, 1, label_trg.long())
        # d_loss_fake = torch.mean(out_fake)
        loss_fake = self.bce_loss(self.zeros_target(self.batch_size), out_fake)

        # Backward and optimize.
        d_loss = loss_real + loss_fake

        # # R1 regularization
        # l1_norm = torch.norm(self.D.weight, p=1)
        # d_loss += l1_norm

        return d_loss, loss_real, loss_fake

    def reset_grad(self):
        """
        Reset the gradient buffers.
        """
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.e_optimizer.zero_grad()
        self.f_optimizer.zero_grad()

    def denorm(self, x):
        """
        Convert the range from [-1, 1] to [0, 1].
        """
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """
        Compute gradient penalty: (L2_norm(dy/dx) - 1)**2.
        """
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """
        Convert label indices to one-hot vectors.
        """
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """
        Generate target domain labels for debugging and testing.
        """
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def generate_style_code(self, label_trg, num_domains=2):
        """

        :param label_trg:
        :return:
        """
        z = self.noise(size=self.batch_size, dimension=16, device=self.device)
        s_tilde = self.F(z, num_domains=num_domains)

        # Compute loss with fake images.
        # s_tilde_trg = torch.index_select(torch.stack(s_tilde, 1), 1, label_trg.squeeze().long())[:, 0, :]
        s_tilde_trg = torch.squeeze(torch.stack([torch.index_select(x, 0, i) for x, i in zip(torch.chunk(torch.stack(s_tilde, 1), chunks=num_domains, dim=1), label_trg.squeeze().long())]))
        return s_tilde_trg

    def classification_loss(self, logit, target, dataset='CelebA'):
        """
        Compute binary or softmax cross entropy loss.
        """
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self):
        """
        Train StarGAN within a single dataset.
        """
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        # c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        # g_lr = self.g_lr
        # d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            # if self.dataset == 'CelebA':
            #     c_org = label_org.clone()
            #     c_trg = label_trg.clone()
            # elif self.dataset == 'RaFD':
            #     c_org = self.label2onehot(label_org, self.c_dim)
            #     c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)  # Input images.
            # c_org = c_org.to(self.device)  # Original domain labels.
            # c_trg = c_trg.to(self.device)  # Target domain labels.
            label_org = label_org.to(self.device)  # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)  # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            # self.d_optimizer.zero_grad()
            # Compute loss with real images.
            # out_src, out_cls = self.D(x_real)
            out_real = self.D(x_real, num_domains=2)

            # d_out_real = torch.gather(out_real, 1, label_org.long())
            # d_loss_real = torch.mean(torch.log(d_out_src))

            # d_loss_real = self.bce_loss(self.zeros_target(self.batch_size), d_out_real)
            # d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)
            #
            # # z:latent code s_tilde:target style code
            # z = self.noise(size=self.batch_size, dimension=16, device=self.device)
            # s_tilde = self.F(z, num_domains=2)
            #
            # # target style code s_tilde
            #
            # # Compute loss with fake images.
            # # s_tilde_tensor = torch.stack(s_tilde, 1)
            # s_tilde_trg = torch.index_select(torch.stack(s_tilde, 1), 1, label_trg.squeeze().long())[:, 0, :]
            # # s_tilde_trg = torch.gather(s_tilde_tensor, 1, label_trg.expand(s_tilde_tensor.size()).long())
            # # s_tilde_trg = torch.gather(torch.stack(s_tilde, 1), 1, torch.unsqueeze(label_trg, 2).long())
            #
            # d_x_fake = self.G(x_real, s_tilde_trg)
            # out_fake = self.D(d_x_fake.detach())
            # d_out_fake = torch.gather(out_fake, 1, label_trg.long())
            # # d_loss_fake = torch.mean(out_fake)
            # d_loss_fake = self.bce_loss(self.ones_target(self.batch_size), d_out_fake)

            # # Compute loss for gradient penalty.
            # alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            # x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            # out_src, _ = self.D(x_hat)
            # d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # # Backward and optimize.
            # d_loss = -(d_loss_real + d_loss_fake)
            #
            # # R1 regularization
            # l1_norm = torch.norm(self.D.weight, p=1)
            # d_loss += l1_norm

            s_tilde_trg = self.generate_style_code(label_trg)

            # d_loss, d_loss_real, d_loss_fake = self.compute_adversarial_loss(True, x_real, label_org, d_x_fake, label_trg)
            d_loss, d_loss_real, d_loss_fake = self.train_discriminator(x_real, label_org, s_tilde_trg, label_trg)

            # d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            # d_loss = -d_loss
            # self.reset_grad()
            # d_loss.backward()
            # self.d_optimizer.step()

            # Logging.
            loss = {
                'D/loss': d_loss.item(),
                'D/loss_real': d_loss_real.item(),
                'D/loss_fake': d_loss_fake.item()
            }

            writer.add_scalar('D/loss', d_loss.item(), i)
            writer.add_scalar('D/loss_real', d_loss_real.item(), i)
            writer.add_scalar('D/loss_fake', d_loss_fake.item(), i)

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                # # style reconstruction
                # g_s_tilde_trg = self.generate_style_code(label_trg)
                # # g_x_fake = self.G(x_real, g_s_tilde_trg)
                # # s_hat = self.E(d_x_fake)
                #
                # # s_hat: estimated style code of source image
                # # loss style reconstruction:style reconstruction
                # s_hat = self.E(self.G(x_real, g_s_tilde_trg), num_domains=2)
                # s_hat_trg = torch.index_select(torch.stack(s_hat, 1), 1, label_trg.squeeze().long())[:, 0, :]
                # g_loss_sty = self.l1_loss(g_s_tilde_trg, s_hat_trg)
                #
                # # loss cycle: preserving source characteristics
                # s_hat_org = torch.index_select(torch.stack(s_hat, 1), 1, label_org.squeeze().long())[:, 0, :]
                # x_fake_cyc = self.G(self.G(x_real, g_s_tilde_trg), s_hat_org)
                # g_loss_cyc = self.l1_loss(x_real, x_fake_cyc)
                #
                # # loss style diversification:style diversification
                # # z1 = self.noise(size=self.batch_size, dimension=16)
                # # s1_tilde = self.F(z1, num_domains=2)
                # # s1_tilde_trg = torch.index_select(torch.stack(s1_tilde, 1), 1, label_trg.squeeze().long())[:, 0, :]
                # s1_tilde_trg = self.generate_style_code(label_trg)
                # # z2 = self.noise(size=self.batch_size, dimension=16)
                # # s2_tilde = self.F(z2, num_domains=2)
                # # s2_tilde_trg = torch.index_select(torch.stack(s2_tilde, 1), 1, label_trg.squeeze().long())[:, 0, :]
                # s2_tilde_trg = self.generate_style_code(label_trg)
                # g_loss_ds = self.l1_loss(self.G(x_real, s1_tilde_trg), self.G(x_real, s2_tilde_trg))

                # # Original-to-target domain.
                # x_fake = self.G(x_real, c_trg)
                # out_src, out_cls = self.D(x_fake)
                # g_loss_fake = - torch.mean(out_src)
                # g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)
                #
                # # Target-to-original domain.
                # x_reconst = self.G(x_fake, c_org)
                # g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # out_real = self.D(x_real, num_domains=2)
                # g_out_real = torch.gather(out_real, 1, label_org.long())
                # g_loss_real = self.bce_loss(self.zeros_target(self.batch_size), g_out_real)
                # # target style code s_tilde
                #
                # # Compute loss with fake images.
                # # x_fake = self.G(x_real, s_tilde[0])
                # out_fake = self.D(g_x_fake)
                # g_out_fake = torch.gather(out_fake, 1, label_org.long())
                # # d_loss_fake = torch.mean(out_fake)
                # g_loss_fake = self.bce_loss(self.ones_target(self.batch_size), g_out_fake)

                g_loss, g_adv_loss, g_loss_sty, g_loss_cyc, g_loss_ds = self.train_generator(x_real, label_org, label_trg)
                # g_loss = self.train_generator(x_real, label_org, label_trg)

                # g_adv_loss = self.compute_adversarial_loss(False, x_real, label_org, g_s_tilde_trg, label_trg)[0]

                # Backward and optimize.
                # g_loss = g_adv_loss + self.lambda_sty * g_loss_sty + self.lambda_cyc * g_loss_cyc + self.lambda_ds * g_loss_ds
                # self.reset_grad()
                # g_loss.backward()
                # self.g_optimizer.step()
                # self.e_optimizer.step()
                # self.f_optimizer.step()

                # Logging.
                # loss['G/loss_fake'] = g_loss_fake.item()
                # loss['G/loss_sty'] = g_loss_sty.item()
                # loss['G/loss_cyc'] = g_loss_cyc.item()
                # loss['G/loss_ds'] = g_loss_ds.item()
                loss['G/loss'] = g_loss.item()
                loss['G/loss_adv'] = g_adv_loss.item()
                loss['G/loss_sty'] = g_loss_sty.item()
                loss['G/loss_cyc'] = g_loss_cyc.item()
                loss['G/loss_ds'] = g_loss_ds.item()

                # writer.add_scalar('G/loss_cyc', g_loss_cyc.item(), i)
                # writer.add_scalar('G/loss_ds', g_loss_ds.item(), i)
                writer.add_scalar('G/loss', g_loss.item(), i)
                writer.add_scalar('G/loss_adv', g_adv_loss.item(), i)
                writer.add_scalar('G/loss_sty', g_loss_sty.item(), i)
                writer.add_scalar('G/loss_cyc', g_loss_cyc.item(), i)
                writer.add_scalar('G/loss_ds', g_loss_ds.item(), i)

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                # if self.use_tensorboard:
                #     for tag, value in loss.items():
                #         self.logger.scalar_summary(tag, value, i + 1)

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    # source images + generated images
                    g_s_tilde_trg = self.generate_style_code(label_trg)
                    x_fake_list = [x_fixed, self.G(x_fixed, g_s_tilde_trg)]
                    # for c_fixed in label_org:
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

                    grid = torchvision.utils.make_grid(x_concat)
                    writer.add_image('images', grid, 0)
                    # writer.add_graph(model, images)

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                E_path = os.path.join(self.model_save_dir, '{}-E.ckpt'.format(i + 1))
                F_path = os.path.join(self.model_save_dir, '{}-F.ckpt'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.E.state_dict(), E_path)
                torch.save(self.F.state_dict(), F_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))
            #
            # # Decay learning rates.
            # if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
            #     g_lr -= (self.g_lr / float(self.num_iters_decay))
            #     d_lr -= (self.d_lr / float(self.num_iters_decay))
            #     self.update_lr(g_lr, d_lr)
            #     print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

            # Decay weight lambda ds
            if (i + 1) < self.num_iters_decay:
                self.lambda_ds = 1 - 0.00002 * (i + 1)
                # print('Decayed weight lambda ds , lambda_ds: {}'.format(self.lambda_ds))

        # close the tensorboard summary writter
        writer.close()

    def train_multi(self):
        """Train StarGAN with multiple datasets."""
        # Data iterators.
        celeba_iter = iter(self.celeba_loader)
        rafd_iter = iter(self.rafd_loader)

        # Fetch fixed inputs for debugging.
        x_fixed, c_org = next(celeba_iter)
        x_fixed = x_fixed.to(self.device)
        c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
        c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
        zero_celeba = torch.zeros(x_fixed.size(0), self.c_dim).to(self.device)  # Zero vector for CelebA.
        zero_rafd = torch.zeros(x_fixed.size(0), self.c2_dim).to(self.device)  # Zero vector for RaFD.
        mask_celeba = self.label2onehot(torch.zeros(x_fixed.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
        mask_rafd = self.label2onehot(torch.ones(x_fixed.size(0)), 2).to(self.device)  # Mask vector: [0, 1].

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for dataset in ['CelebA', 'RaFD']:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch real images and labels.
                data_iter = celeba_iter if dataset == 'CelebA' else rafd_iter

                try:
                    x_real, label_org = next(data_iter)
                except:
                    if dataset == 'CelebA':
                        celeba_iter = iter(self.celeba_loader)
                        x_real, label_org = next(celeba_iter)
                    elif dataset == 'RaFD':
                        rafd_iter = iter(self.rafd_loader)
                        x_real, label_org = next(rafd_iter)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if dataset == 'CelebA':
                    c_org = label_org.clone()
                    c_trg = label_trg.clone()
                    zero = torch.zeros(x_real.size(0), self.c2_dim)
                    mask = self.label2onehot(torch.zeros(x_real.size(0)), 2)
                    c_org = torch.cat([c_org, zero, mask], dim=1)
                    c_trg = torch.cat([c_trg, zero, mask], dim=1)
                elif dataset == 'RaFD':
                    c_org = self.label2onehot(label_org, self.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.c2_dim)
                    zero = torch.zeros(x_real.size(0), self.c_dim)
                    mask = self.label2onehot(torch.ones(x_real.size(0)), 2)
                    c_org = torch.cat([zero, c_org, mask], dim=1)
                    c_trg = torch.cat([zero, c_trg, mask], dim=1)

                x_real = x_real.to(self.device)  # Input images.
                c_org = c_org.to(self.device)  # Original domain labels.
                c_trg = c_trg.to(self.device)  # Target domain labels.
                label_org = label_org.to(self.device)  # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)  # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org, dataset)

                # Compute loss with fake images.
                x_fake = self.G(x_real, c_trg)
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i + 1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg, dataset)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training info.
                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(et, i + 1, self.num_iters, dataset)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i + 1)

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_celeba_list:
                        c_trg = torch.cat([c_fixed, zero_rafd, mask_celeba], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    # for c_fixed in c_rafd_list:
                    #     c_trg = torch.cat([zero_celeba, c_fixed, mask_rafd], dim=1)
                    #     x_fake_list.append(self.G(x_fixed, c_trg))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.celeba_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
                c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
                zero_celeba = torch.zeros(x_real.size(0), self.c_dim).to(self.device)  # Zero vector for CelebA.
                zero_rafd = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)  # Zero vector for RaFD.
                mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
                mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)  # Mask vector: [0, 1].

                # Translate images.
                x_fake_list = [x_real]
                for c_celeba in c_celeba_list:
                    c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                for c_rafd in c_rafd_list:
                    c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

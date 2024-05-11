import math
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from skimage import img_as_ubyte
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import os


def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= args.batch_size  
  return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def pred(validate_seq, validate_cond, modules, args, device):
    validate_seq = validate_seq.to(device)
    validate_cond = validate_cond.to(device)
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    gen_seq = []
    gen_seq.append(validate_seq[0])
    x_in = validate_seq[0]
    h_seq = [modules['encoder'](validate_seq[i]) for i in range(args.n_past+args.n_future)]
    for i in range(1, args.n_past+args.n_future):
        h_target = h_seq[i][0].detach()
        if args.last_frame_skip or i < args.n_past:	
            h, skip = h_seq[i-1]
        else:
            h, _ = h_seq[i-1]
        h = h.detach()
        
        if i < args.n_past:
            z_t, _, _ = modules['posterior'](h_seq[i][0])
            modules['frame_predictor'](torch.cat([validate_cond[i-1], h, z_t], 1)) 
            gen_seq.append(validate_seq[i])
        else:
            z_t = torch.randn(args.batch_size, args.z_dim).cuda()
            h_pred = modules['frame_predictor'](torch.cat([validate_cond[i-1], h, z_t], 1)).detach()
            x_pred = modules['decoder']([h_pred, skip]).detach()
            h_seq[i] = modules['encoder'](x_pred)
            gen_seq.append(x_pred)
    gen_seq = torch.stack(gen_seq)
    return gen_seq

def plot_pred(validate_seq, validate_cond, modules, epoch, args, device, sample_idx=0):
    pred_seq = pred(validate_seq, validate_cond, modules, args, device)

    print("[Epoch {}] Saving predicted images & GIF...".format(epoch))
    os.makedirs("{}/gen/epoch-{}-pred".format(args.log_dir, epoch), exist_ok=True)
    gt_images, images, pred_frames, gt_frames = [], [], [], []
    sample_seq, gt_seq = pred_seq[:, sample_idx, :, :, :], validate_seq[:, sample_idx, :, :, :]
    for frame_idx in range(sample_seq.shape[0]):
        img_file = "{}/gen/epoch-{}-pred/{}.png".format(args.log_dir, epoch, frame_idx)
        gt_img_file = "{}/gen/epoch-{}-pred/{}_gt.png".format(args.log_dir, epoch, frame_idx)
        save_image(sample_seq[frame_idx], img_file)
        save_image(gt_seq[frame_idx], gt_img_file)
        images.append(imageio.imread(img_file))
        gt_images.append(imageio.imread(gt_img_file))
        pred_frames.append(sample_seq[frame_idx])
        os.remove(img_file)
        os.remove(gt_img_file)

        gt_frames.append(gt_seq[frame_idx])

    pred_grid = make_grid(pred_frames, nrow=sample_seq.shape[0])
    gt_grid   = make_grid(gt_frames  , nrow=gt_seq.shape[0])
    save_image(pred_grid, "{}/gen/epoch-{}-pred/pred_grid.png".format(args.log_dir, epoch))
    save_image(gt_grid  , "{}/gen/epoch-{}-pred/gt_grid.png".format(args.log_dir, epoch))
    imageio.mimsave("{}/gen/epoch-{}-pred/animation.gif".format(args.log_dir, epoch), images)
    imageio.mimsave("{}/gen/epoch-{}-pred/gt_animation.gif".format(args.log_dir, epoch), gt_images)


def plot_kl_curve(x, kl_losses, kl_betas, tfrs, title, log_dir):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    cmap = plt.get_cmap("tab10")

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('KL loss')
    ax2.set_ylabel("Teacher Forcing Ratio / KL Anneal Beta")  # we already handled the x-label with ax1
    ax1.set_ylim((0.0, 0.1))

    h1, = ax1.plot(x, kl_losses, color=cmap(0), label="KL loss")
    h2, = ax2.plot(x, kl_betas, color=cmap(1), linestyle="dotted", label="KL Anneal Beta")
    h3, = ax2.plot(x, tfrs, color=cmap(2), linestyle="dashed", label="Teacher Forcing Ratio")

    ax1.legend(handles=[h1, h2, h3], loc="best")

    plt.title(title)
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('{}/KL_loss.png'.format(log_dir))

def plot_psnr_curve(x, psnrs, title, log_dir):
    plt.subplots(1)
    plt.plot(x, psnrs)
    plt.xlabel("Epoch"), plt.ylabel("PSNR")

    plt.title(title)
    plt.tight_layout()
    plt.savefig('{}/psnr.png'.format(log_dir))


def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return torchvision.utils.save_image(images, filename)

def image_tensor(inputs, padding=1):
    assert len(inputs) > 0

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result
    
    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))

def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)

def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    # pdb.set_trace()
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)

def save_gif(filename, inputs, duration=0.25):
    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1)
        images.append(img.numpy())
    imageio.mimsave(filename, img_as_ubyte(images), duration=duration)

def save_gif_with_text(filename, inputs, text, duration=0.25):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1).numpy()
        images.append(img)
    imageio.mimsave(filename, img_as_ubyte(images), duration=duration)

def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0,0,0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)
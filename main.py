import warnings
import datetime
from medpy.io import load
from skimage.io import imsave

from dataset import Multi_PNGDataset,Multi_PNGDataset_val
from ini_file_io import load_train_ini
from progress_bar import ProgressBar
from network.resnet import *
from seg_eval import *
from utils import *

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchsummary import summary
from torchvision.transforms import ToPILImage

from tensorboardX import SummaryWriter

# set cuda visable device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_GPU else "cpu")
num_workers = 8

image_transform = ToPILImage()

def train(param_set,model):
    folder = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    save_dir = param_set['result_dir'] + folder
    ckpt_dir = save_dir + '/checkpoint'
    log_dir = save_dir + '/log'
    test_result_dir = save_dir + '/testResult'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.mkdir(ckpt_dir)
        os.mkdir(test_result_dir)
    for file in os.listdir(log_dir):
        print('removing ' + os.path.join(log_dir, file))
        os.remove(os.path.join(log_dir, file))

    test_batch = ['batch5']
    val_loader = DataLoader(Multi_PNGDataset_val(param_set['imgdir'],test_batch),num_workers=num_workers, batch_size=param_set['batch_size'], shuffle=False)
    gt_labels = get_gt_labels(os.path.join(param_set['imgdir'], test_batch[0], 'mask/'))  # when evaluate
    '''c_weight = torch.ones(NUM_CLASSES)
    if USE_GPU:
        criterion = CrossEntropyLoss2d(c_weight.cuda())  # define the criterion'''
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),lr = 5e-4)  # define the optimizer
    # lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 200], gamma=0.1)  # lr decay
    # lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)  # for debug

    epoch_save = param_set['epoch_save']
    num_epochs = param_set['epoch']
    cvBatch = ['batch4', 'batch3', 'batch2', 'batch1']
    writer = SummaryWriter(log_dir)
    iter_count = 0
    #writer.add_graph(model, )
    # loader = DataLoader(PNGDataset(param_set['imgdir'],cvBatch),num_workers=num_workers, batch_size=param_set['batch_size'], shuffle=True)
    loader = DataLoader(Multi_PNGDataset(param_set['imgdir'],cvBatch),num_workers=num_workers, batch_size=param_set['batch_size'], shuffle=True)
    print('steps per epoch:', len(loader))
    best_val_dice = 0.5  # 0 for debug
    model.train()
    for epoch in range(num_epochs+1):
        # lr_schedule.step()   # lr decay
        train_progressor = ProgressBar(mode='Train', epoch=epoch, total_epoch=num_epochs, model_name=param_set['model'],total = len(loader))
        # train dataloader
        iter_loss = 0
        iter_dice = 0
        for step, (images_pv, images_art,labels) in enumerate(loader):
            train_progressor.current = step
            model.train()
            if USE_GPU:
                images_pv = images_pv.cuda(non_blocking=True)  # inputs to GPU
                images_art = images_art.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            outputs = model(images_pv,images_art)  # forward
            out_soft = F.softmax(outputs, dim=1).data.cpu()
            out_dice = soft_dice(out_soft[:, 1].cpu(), labels.cpu().float())

            iter_dice += out_dice.cpu()
            loss = criterion(outputs, labels)  # loss

            iter_loss += float(loss.item())

            writer.add_scalar('train/loss', loss.item(), epoch * len(loader) + step)
            # writer.add_image('train/input_pv',images_pv[0],global_step=10)
            # writer.add_image('train/input_art',images_art[0],global_step=10)
            # writer.add_image('train/gt',labels[0],global_step = 10)
            # writer.add_image('train/output',out_soft[0,1,:,:],global_step=10)

            train_progressor.current_loss = loss.item()
            train_progressor.current_dice = out_dice.cpu()

            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backward
            optimizer.step()  # optimize
            train_progressor()

            iter_count += 1
            # clear cache
            del images_pv, images_art, labels, loss, outputs, out_soft, out_dice
            # import gc
            # gc.collect()
            torch.cuda.empty_cache()

        train_progressor.done()
        # save best model in terms of validation dice---------
        #evaluate
        valid_result = evaluate(val_loader,model,criterion,epoch,gt_labels)
        # print("epoch {epoch}, validation dice {val_dice}".format(epoch=epoch, val_dice=valid_result[0]))  #for debug
        with open(save_dir + "/validation_log.txt", "a") as f:
            print("epoch {epoch}, validation dice {val_dice}".format(epoch=epoch, val_dice=valid_result[0]), file=f)

        if valid_result[0]>=0.5:
            is_best = valid_result[0] > best_val_dice
            best_val_dice = max(valid_result[0],best_val_dice)
            filename = "{model}-{epoch:03}-{step:04}-{dice}.pth".format(model=param_set['model'], epoch=epoch,
                                                                 step=step,dice=valid_result[0])

            #save_checkpoint(model.state_dict(), is_best, ckpt_dir, filename)
            save_checkpoint_new(model.state_dict(), is_best, ckpt_dir, filename, SAVE=(epoch>0 and epoch % epoch_save == 0 or epoch==num_epochs))

        epoch_loss = iter_loss/len(loader)
        epoch_dice = iter_dice/len(loader)
        writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
        writer.add_scalar('train/epoch_dice', epoch_dice, epoch)

    writer.close()

def evaluate(val_loader, model, criterion, epoch, gt_labels):
    #progress bar
    val_progressor = ProgressBar(mode='Val',epoch=epoch,total_epoch=param_set['epoch'],model_name=param_set['model'],total=len(val_loader))
    #switch to evaluate model and confirm model has been transferred to cuda
    model.cuda()
    model.eval()
    with torch.no_grad():
        iter_loss = 0
        iter_dice = 0
        val_iter_loss = 0
        val_iter_dice = 0
        pred_labels = []
        for step, (images_pv,images_art, labels) in enumerate(val_loader):
            val_progressor.current = step
            if USE_GPU:
                images_pv = images_pv.cuda()
                images_art = images_art.cuda()
                labels = labels.cuda()
            #compute the output
            outputs = model(images_pv,images_art)
            val_loss = criterion(outputs, labels)
            prob = F.softmax(outputs, dim=1).data.cpu()
            val_out_soft = F.softmax(outputs,dim=1).data.cpu()
            val_dice = soft_dice(val_out_soft[:,1],labels.cpu().float())
            val_iter_loss += float(val_loss.item())
            val_iter_dice += val_dice.cpu()
            pred_labels.append(prob[:,1])
            val_progressor.current_loss = val_loss.item()
            val_progressor.current_dice = val_dice.cpu()
            val_progressor()

            # clear cache
            del images_pv, images_art, labels, outputs, val_loss, val_dice
            torch.cuda.empty_cache()
        val_progressor.done()
        # val_epoch_loss = sum(val_iter_loss) / len(loader)
        # val_epoch_dice = sum(val_iter_dice) / len(loader)
        # arr_pred_labels = np.asarray(pred_labels)   # np.array
        arr_pred_labels = torch.cat(pred_labels, dim=0)
        arr_pred_labels[arr_pred_labels < 0.6] = 0
        arr_pred_labels[arr_pred_labels >= 0.6] = 1
        global_dice = soft_dice(arr_pred_labels, gt_labels)
        val_epoch_loss = val_iter_loss / len(val_loader)
        val_epoch_dice = val_iter_dice / len(val_loader)
        del pred_labels, arr_pred_labels
        return [global_dice, val_epoch_loss, val_epoch_dice]

def predict(param_set,model):
    ckpt_dir = param_set['model_loc']
    folder = param_set['folder']
    save_dir = param_set['result_dir'] + folder

    dice_dir = save_dir + '/dice'
    test_result_dir = save_dir + '/testResult'
    test_batch = ['batch5']
    pv_test_dir = param_set['testdir']
    art_test_dir = pv_test_dir.replace('PV', 'ART')

    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)
    if not os.path.exists(dice_dir):
        os.mkdir(dice_dir)

    # load model
    model_name = ckpt_dir.split('/')[-1]
    model.load_state_dict(torch.load(ckpt_dir))
    model.eval()
    print('Current model: ',model_name)

    model_test_dir = os.path.join(test_result_dir,model_name.split('.')[0])
    # save
    if not os.path.exists(model_test_dir):
        os.mkdir(model_test_dir)

    print('Applying on: ' + test_batch[0])
    pred_labels = []
    gt_labels = []
    all_dice = []
    sliceIdxs = []
    print('evaluate on test images...')
    #f = open(os.path.join(dice_dir, model_name + '_global_dice.txt'), 'a')
    dice_file_name = model_name.split('.')[0] + '.txt'
    f_dice = open(os.path.join(dice_dir, dice_file_name), 'w')
    files = os.listdir(os.path.join(pv_test_dir,'img'))
    files.sort()
    pre_case = files[0][3:6]
    for file in files:
        if file.endswith('.png'):
            cur_case = file[3:6]
            if cur_case != pre_case:
                liver_mask = get_liver_mask(pre_case, min(sliceIdxs), max(sliceIdxs))
                arr_pred_labels = np.array(pred_labels)
                arr_gt_labels = np.array(gt_labels)
                arr_pred_labels[liver_mask==0]=0
                arr_pred_labels[arr_pred_labels>=0.6]=1
                k_dice = dice_coef(arr_pred_labels, arr_gt_labels)
                f_dice.write(pre_case + str(k_dice)+'\n')
                all_dice.append(k_dice)
                print('Case {caseNum} test done,dice: {dice}'.format(caseNum=pre_case, dice=k_dice))
                # reset
                pred_labels = []
                gt_labels = []
                sliceIdxs = []
                pre_case = cur_case

            with open(os.path.join(pv_test_dir, 'img', file), 'rb') as f:
                image_pv = load_image(f).convert('RGB')
            with open(os.path.join(art_test_dir, 'img', file), 'rb') as f:
                image_art = load_image(f).convert('RGB')
            with open(os.path.join(pv_test_dir, 'mask', file), 'rb') as f:
                label = load_image(f).convert('L')
            sliceIdxs.append(int(file.split('_')[1][-3:]))
            image_pv = np.asarray(image_pv, np.float32) / 255.0
            image_art = np.asarray(image_art, np.float32) / 255.0
            label = np.asarray(np.array(label) / 255, dtype=np.uint8)
            # pmp_img = model_apply(model, image_pv, image_art, param_set['ins'],file.split('.')[0])  #224
            pmp_img = model_apply_full(model,image_pv,image_art)  #512
            output = Image.fromarray(np.uint8(pmp_img)).convert('L')
            out_str = model_test_dir + '/'+ file
            #output.save(out_str)
            imsave(out_str, np.asarray(pmp_img * 255., np.uint8))
            pred_labels.append(pmp_img)
            gt_labels.append(label)
    liver_mask = get_liver_mask(pre_case, min(sliceIdxs), max(sliceIdxs))
    arr_pred_labels = np.array(pred_labels)
    arr_gt_labels = np.array(gt_labels)
    arr_pred_labels[liver_mask == 0] = 0
    arr_pred_labels[arr_pred_labels >= 0.6] = 1
    k_dice = dice_coef(arr_pred_labels, arr_gt_labels)
    f_dice.write(pre_case + str(k_dice) + '\n')
    all_dice.append(k_dice)
    print('Case {caseNum} test done,dice: {dice}'.format(caseNum=pre_case, dice=k_dice))
    mean_dice = np.mean(np.array(all_dice), axis=0)
    print('mean dice: {dice}'.format(dice=mean_dice))
    f_dice.write(str(mean_dice)+'\n')
    f_dice.close()

def get_liver_mask(pre_case,minz,maxz):
    # load predicted liver mask
    gt_path = '/Project/Data/PV/'
    mask, mask_header = load(os.path.join(gt_path, 'segmentation-' + pre_case + '.nii'))
    liver_mask = np.zeros(mask.shape)
    liver_mask[np.where(mask != 0)] = 1
    liver_mask = np.transpose(liver_mask, (2, 1, 0))
    liver_mask = liver_mask[minz:maxz + 1, :, :]
    return liver_mask

def model_apply_full(model,image_pv, image_art):
    tI_pv = np.transpose(image_pv,[2,0,1])
    tI_pv = torch.Tensor(tI_pv.copy()).cuda()
    tI_art = np.transpose(image_art, [2, 0, 1])
    tI_art = torch.Tensor(tI_art.copy()).cuda()
    pred = model(tI_pv.unsqueeze(0),tI_art.unsqueeze(0))
    prob = F.softmax(pred,dim=1).squeeze(0).data.cpu().numpy()
    # prob = F.softmax(pred,dim=1).data.cpu().numpy()
    pmap_img = prob[1]
    return pmap_img

def model_apply(model,image_pv,image_art,ins,file):
    avk = 4
    nrotate = 1
    wI = np.zeros([ins, ins])
    pmap = np.zeros([image_pv.shape[0], image_pv.shape[1]])
    avI = np.zeros([image_pv.shape[0], image_pv.shape[1]])
    for i in range(ins):
        for j in range(ins):
            dx = min(i, ins - 1 - i)
            dy = min(j, ins - 1 - j)
            d = min(dx, dy) + 1
            wI[i, j] = d
    wI = wI / wI.max()

    for i1 in range(math.ceil(float(avk) * (float(image_pv.shape[0]) - float(ins)) / float(ins)) + 1):
        for j1 in range(math.ceil(float(avk) * (float(image_pv.shape[1]) - float(ins)) / float(ins)) + 1):

            # start and end index
            insti = math.floor(float(i1) * float(ins) / float(avk))
            instj = math.floor(float(j1) * float(ins) / float(avk))
            inedi = insti + ins
            inedj = instj + ins

            small_pmap = np.zeros([ins, ins])

            for i in range(nrotate):
                small_in_pv = image_pv[insti:inedi, instj:inedj]
                small_in_art = image_art[insti:inedi, instj:inedj]
                small_in_pv = np.rot90(small_in_pv,i)
                small_in_art = np.rot90(small_in_art,i)

                tI_pv = np.transpose(small_in_pv,[2,0,1])
                tI_art = np.transpose(small_in_art,[2,0,1])
                tI_pv = torch.Tensor(tI_pv.copy()).cuda()
                tI_art = torch.Tensor(tI_art.copy()).cuda()
                pred = model(tI_pv.unsqueeze(0),tI_art.unsqueeze(0))

                prob = F.softmax(pred,dim=1).squeeze(0).data.cpu().numpy()
                small_out = prob[1]
                small_out = np.rot90(small_out,-i)

                small_pmap = small_pmap + np.array(small_out)

            small_pmap = small_pmap / nrotate

            pmap[insti:inedi, instj:inedj] += np.multiply(small_pmap, wI)
            avI[insti:inedi, instj:inedj] += wI
    pmap_img = np.divide(pmap, avI)
    return pmap_img


def load_image(file):
    return Image.open(file)

def main(param_set):

    warnings.filterwarnings('ignore')
    print('====== Phase >>> %s <<< ======' % param_set['mode'])
    NUM_CHANNELS = param_set['numChannels']  # number of input channels
    NUM_CLASSES = param_set['nclass']  # number of class
    ins = 224
    nc = 64
    model = None
    if param_set['model'] == 'Resnet_2d':
        model = RESNET_2D(NUM_CLASSES,NUM_CHANNELS,nc)
    elif param_set['model'] == 'RESNET_phase_att':
        model = RESNET_phase_att(NUM_CLASSES,NUM_CHANNELS,nc)

    print('model: ',param_set['model'])

    print(DEVICE)
    model.to(DEVICE)  # send to GPU
    pre_train = False
    if param_set['mode']=='train':
        if pre_train:
            ckpt_dir = param_set['model_loc']
            model.load_state_dict(torch.load(ckpt_dir))
            model_name = ckpt_dir.split('/')[-1]
            print('load pretrained model ', model_name)
        train(param_set,model)
    elif param_set['mode'] == 'test':
        predict(param_set,model)
    elif param_set['mode'] == 'eval_all':
        ckpt_path = param_set['ckpt_path']
        ckpts = os.listdir(ckpt_path)
        for ckpt in ckpts:
            param_set['model_loc'] = os.path.join(ckpt_path,ckpt)
            predict(param_set,model)

if __name__ == '__main__':
    # load training parameter #
    ini_file = '/Project/PA-ResSeg/tr_param.ini'
    param_sets = load_train_ini(ini_file)
    param_set = param_sets[0]
    main(param_set)



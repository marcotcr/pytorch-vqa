import h5py
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import model

import config
import data
import utils
from resnet import resnet as caffe_resnet
from PIL import Image
import matplotlib.pyplot as plt

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = caffe_resnet.resnet152(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer

class VQAModel:
    def __init__(self, model_path='/home/marcotcr/phd/pytorch-vqa/2017-08-04_00.55.19.pth', max_question_length=23):
        cudnn.benchmark = True
        self.resnet = ResNet().cuda()
        self.resnet.eval()
        log = torch.load(model_path)
        tokens = len(log['vocab']['question']) + 1
        self.token_to_index = log['vocab']['question']
        self.answer_to_index = log['vocab']['answer']
        index_to_answer = {v: k for k, v in self.answer_to_index.items()}
        self.label_names = [index_to_answer[i] for i in sorted(index_to_answer)]
        self.net = torch.nn.DataParallel(model.Net(tokens))
        self.net.load_state_dict(log['weights'])
        self.net.eval()
        self.max_question_length = max_question_length

    def load_image(self, path, show=False):
        transform = utils.get_transform(config.image_size, config.central_fraction)
        img = Image.open(path).convert('RGB')
        if show:
            plt.figure()
            plt.imshow(img)
        if transform is not None:
            img = transform(img)
        return img

    def imgs_to_features(self, imgs):
        with torch.no_grad():
            imgs = Variable(imgs.cuda(async=True))#, volatile=True)
            out = self.resnet(imgs)
        return out

    def paths_to_features(self, paths):
        if not len(paths):
            return []
        imgs = []
        prev = ''
        img = ''
        for path in paths:
            if path != prev:
                img = self.load_image(path)
            imgs.append(img)
        imgs = torch.stack(imgs)
        return self.imgs_to_features(imgs)

    def encode_questions(self, questions):
        """ Turn a question into a vector of indices and a question length """
        vec = torch.zeros(len(questions), self.max_question_length).long()
        lens = []
        for j, question in enumerate(questions):
            for i, token in enumerate(question):
                index = self.token_to_index.get(token, 0)
                vec[j, i] = index
            lens.append(len(question))
        return vec, np.array(lens)

    def encode_answers(self, answers):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        answer_vec = torch.zeros(len(answers), len(self.answer_to_index))
        for i, answers_ in enumerate(answers):
            for answer in answers_:
                index = self.answer_to_index.get(answer)
                if index is not None:
                    answer_vec[i, index] += 1
        return answer_vec

    def predict_proba_given_image_features(self, questions, image_features):
        """questions and image_features must have same length"""
        q, q_len = self.encode_questions([q.lower().split() for q in questions])
        idxs = list(np.argsort(q_len)[::-1])
        q = q[idxs]
        q_len = q_len[idxs]
        image_features = image_features[idxs]
        q_len = torch.from_numpy(q_len)
        softmax = nn.Softmax(1).cuda()
        with torch.no_grad():
            var_params = {
                # 'volatile': True,
                'requires_grad': False,
            }
            v = Variable(image_features.cuda(async=True), **var_params)
            q = Variable(q.cuda(async=True), **var_params)
        #     a = Variable(a.cuda(async=True), **var_params)
            q_len = Variable(q_len.cuda(async=True), **var_params)
            out = self.net(v, q, q_len)
            predict_proba = softmax(out)
        return predict_proba.cpu().numpy()[idxs]

    def predict_proba_same_image(self, questions, image_features, batch_size=16):
        """image_features is the set of features for a single image, 3d array"""
        chunked = chunks(questions, batch_size)
        image_features = image_features.repeat(batch_size, 1, 1, 1)
        all_pp = []
        for x in chunked:
            if len(x) < batch_size:
                image_features = image_features[:len(x)]
            pp = self.predict_proba_given_image_features(x, image_features)
            all_pp.append(pp)
        return np.vstack(all_pp)
        pass

    def predict_proba_from_paths(self, questions, image_paths, batch_size=16):
        zipped = list(zip(questions, image_paths))
        chunked = chunks(zipped, batch_size)
        all_pp = []
        for x in chunked:
            qs, ps= zip(*x)
            v = self.paths_to_features(ps)
            pp = self.predict_proba_given_image_features(qs, v)
            all_pp.append(pp)
        return np.vstack(all_pp)

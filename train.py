import numpy as np
import dezero
import dezero.functions as F
import dezero.layers as L
import dezero.models as M
from dezero import cuda
from dezero.models import Model


from dezero import cuda
use_gpu = cuda.gpu_enable
use_gpu = False
print(use_gpu)

from dezero import cuda
from typing import List, Optional, Tuple, Any
import math

class StrokesDataset(dezero.DataLoader):
    def __init__(self, data, batch_size, max_seq_length: int, scale: Optional[float] = None, shuffle=True, gpu=False):
        stroke_data = []
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(data)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu
        
        xp = cuda.cupy if self.gpu else np
        
        for seq in data:
            # we will deem a sequence that is less than 10 as too short and thus ignore it
            if 10 < len(seq) <= max_seq_length:
                # clamp the delta x and delta y to [-1000, 1000]
                seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                
                seq = np.array(seq, dtype=np.float32)
                stroke_data.append(seq)
        
        if scale is None:
            # calculate the scale factor
            # the scale factor is the standard deviation of the x and y coordinates
            # mean is not adjusted for simplicity
            # 0:2 means the first two columns of the array which are x and y coordinates
            scale = np.std(np.concatenate([np.ravel(s[:,0:2]) for s in stroke_data]))
        
        longest_seq_len = max([len(seq) for seq in stroke_data])
        
        # we add two extra columns to the dataset since we currently there are only 3 columns in the dataset
        # additional two columns are for changing the last point 1/0 to a one-hot vector
        temp_stroke_dataset = xp.zeros((len(stroke_data), longest_seq_len + 2, 5), dtype=np.float32)
        
        # self.mask is used to mark areas of the sequence that are not used
        # we first initialize it to zero
        temp_mask_dataset = xp.zeros((len(stroke_data), longest_seq_len + 1))
        
        self.dataset = []
        
        # start of sequence is [0, 0, 1, 0, 0]
        
        for i, seq in enumerate(stroke_data):
            seq = xp.array(seq, dtype=xp.float32)
            len_seq = len(seq)
            
            # we start from 1 to leave the first row for the start of sequence token
            temp_stroke_dataset[i, 1:len_seq + 1, 0:2] = seq[:, :2] / scale # this is the x and y coordinates
            temp_stroke_dataset[i, 1:len_seq + 1, 2] = 1 - seq[:, 2] # this is the pen down
            temp_stroke_dataset[i, 1:len_seq + 1, 3] = seq[:, 2] # this is the pen up
            temp_stroke_dataset[i, len_seq + 1, 4] = 1  # this is the end of sequence token
            temp_mask_dataset[i, :len_seq + 1] = 1 # mask is on until the end of the sequence 
            # self.mask is used to mark areas of the sequence that are not used
            # for example, if the sequence is shorter than the longest sequence, we use mask to ignore the rest of the sequence
            # an example of mask is [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        temp_stroke_dataset[:, 0, 2] = 1
        
        for i in range(len(stroke_data)):
            self.dataset.append([temp_stroke_dataset[i], temp_mask_dataset[i]])
        
        
        self.reset()

import dezero.functions as F

# According to other estimates
# the number of distributions in the mixture model is 20
# https://github.com/Shun14/sketch-rnn-kanji
# https://nn.labml.ai/sketch_rnn/index.html

# This is for getting the loss of delta_x and delta_y
class BivariateGaussianMixture:
    def __init__(self, pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
        # check if the pi_probs sum for each sequence is 1
        # print('test pi', test_pi.shape) # pi shape is (batch_size, seq_len, n_distributions)
        # check if the pi probabilities sum to 1
        # seq_len = pi_logits.shape[1]
        # print(F.reshape(F.sum(pi_logits, axis=2), (-1, seq_len)), 'sum of pi')
        self.pi_logits = pi_logits
        self.pi_probs = F.softmax(pi_logits, axis=2)
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho_xy = rho_xy
    
    @property
    def n_distributions(self):
        return self.pi_logits.shape[-1]
    
    def set_temperature(self, temperature: float):
        self.pi_logits /= temperature
        self.pi_probs = F.softmax(self.pi_logits, axis=2) # we do this to make sure the pi probabilities sum to 1
        self.sigma_x *= math.sqrt(temperature)
        self.sigma_y *= math.sqrt(temperature)
    
    def gaussian_pdf(self, x_delta, y_delta):
        # the result means the probability of y in the normal distribution
        # we check the probability of y in the normal distribution
        # if the probability is high, the result is close to 1
        # x_delta and y_delta shape are (batch_size, seq_len, hidden_size)
        # mu_x and mu_y shape are (batch_size, seq_len, n_distributions)
        norm1 = F.sub(x_delta, self.mu_x)
        norm2 = F.sub(y_delta, self.mu_y)
        xp = cuda.get_array_module(norm1)

        dtype = self.sigma_x.dtype
        max_dtype = xp.finfo(dtype).max
        self.sigma_x = F.clip(self.sigma_x, 1e-5, max_dtype)
        self.sigma_y = F.clip(self.sigma_y, 1e-5, max_dtype)
        self.rho_xy = F.clip(self.rho_xy, -1 + 1e-5, 1 - 1e-5)
        
        s1s2 = F.mul(self.sigma_x, self.sigma_y)
        
        # This is from: https://github.com/hardmaru/write-rnn-tensorflow/blob/master/model.py
        # z = tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2))
        #     - 2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)
         
        # below is the deconstruction of the above linez
        z_first_term = F.pow(F.div(norm1, self.sigma_x), 2)
        z_second_term = F.pow(F.div(norm2, self.sigma_y), 2)
        z_last_term_inner = F.mul(self.rho_xy, F.mul(norm1, norm2))
        z_last_term_middle = F.div(z_last_term_inner, s1s2)
        tmp_z = xp.ones(z_last_term_middle.shape) * -2
        z_last_term = F.mul(tmp_z, z_last_term_middle)
        z = F.add(F.add(z_first_term, z_second_term), z_last_term)
        negRho = F.sub(np.ones(self.rho_xy.shape), F.pow(self.rho_xy, 2))

        
        result = F.exp(F.div(-z, 2 * negRho))
        deno_first_term = np.ones(self.sigma_x.shape) * 2 * math.pi
        denom_second_term = F.mul(s1s2, F.pow(negRho, 0.5))
        denom = F.mul(deno_first_term, denom_second_term)
        result = F.div(result, denom)
        
        return result

    # x1_data and x2_data are the real x and y coordinates of the stroke
    def get_lossfunc(self, x_delta, y_delta, mask):
        result0 = self.gaussian_pdf(x_delta, y_delta)
        # check if result0 has inf or nan
        # result0 shape is (batch_size, seq_len, n_distributions) 3D
        result1 = F.mul(result0, self.pi_logits) # pi_logits shape is (batch_size, seq_len, n_distributions)
        
        result1 = F.sum(result1, axis=2, keepdims=True) # sum over the distributions
        # the result1 shape is (batch_size, seq_len, 1)
        # we reshape it to (batch_size, seq_len)
        result1 = F.reshape(result1, result1.shape[:-1]) # result.shape[:-1] is (batch_size, seq_len)
        
        dtype = result1.dtype
        max_dtype = np.finfo(dtype).max
        result1 = F.clip(result1, 1e-5, max_dtype) 

        result1 = -F.log(result1) # result1 shape is (batch_size, seq_len)
        # mask shape is also (batch_size, seq_len)
        result1 = F.mul(result1, mask) # we multiply the mask to ignore the padding
        
        
        # make the value to be one number
        
        
        return F.mean(result1)
    
    def get_pi_idx(self, x, out_pi_elem):
        # pdf shape is (batch_size, seq_len, n_distributions)
        # let us only get the first batch
        pdf = out_pi_elem
        N = pdf.size
        accumulate = 0
        # print(pdf.size, 'pdf size')
        # print(F.sum(pdf), 'sum of pdf')
        # print(out_pi_elem.shape, 'out_pi_elem shape')
        # print(x, 'x in pi idx', pdf, " pdf", pdf.shape)
        # print("hello",pdf[0], x)
        for i in range(0, N):
            # print(pdf[i].data, 'pdf[i].data')
            accumulate += pdf[i].data
            if accumulate >= x:
                return i
            # print(accumulate, 'accumulate')
        print('error with sampling ensemble')
        return -1
    
    # M means the number of samples
    def sample(self, count, M=15):
        xp = cuda.get_array_module(self.pi_logits)
        # get the index of the distribution
        
        result = xp.random.rand(count, M, 3) # initially random [0,1]
        # print(result.shape, 'result shape')
        # we will get result for delta_x and delta_y
        rn = xp.random.rand(count, M, 2) 
        mu = 0
        std = 0
        idx = 0
        
        # currently the pi logits shape is (batch_size, seq_len, n_distributions)
        # we will only get the first batch for now'
        # print(self.pi_logits.shape, 'pi logits shape')
        # print(self.pi_logits[0].shape, 'pi logits shape')
        # print("get sum of pi", F.sum(self.pi_logits[0]))
        out_pi = self.pi_probs[0] # out_pi shape is (seq_len, n_distributions)
        # print(out_pi.shape, 'out_pi shape')
        # print("mu_x", self.mu_x.shape)
        # print("std", self.sigma_x.shape)
        
        # we do not need to get batch size since we are only getting the first batch
        mu_x = self.mu_x[0]
        mu_y = self.mu_y[0]
        sigma_x = self.sigma_x[0]
        sigma_y = self.sigma_y[0]
        rho_xy = self.rho_xy[0]
        
        for j in range(M):
            for i in range(count):
                # we only get the first element since we only need one
                idx = self.get_pi_idx(result[i, j, 0], out_pi[i])
                mu = [mu_x[i, idx], mu_y[i, idx]]
                std = [sigma_x[i, idx], sigma_y[i, idx]]
                rho = rho_xy[i, idx]
                
            
                # print(mu + rn[i, j] * std, 'mu + rn[i, j] * std')
                result_x_y = (mu + rn[i, j] * std)
                # print(result_x_y[0].data, 'this is resuult')
                result[i, j, 0] = result_x_y[0].data
                result[i, j, 1] = result_x_y[1].data
        return result

class Encoder(Model):
    def __init__(self, d_z, hidden_size):
        super().__init__()
        self.lstm = L.LSTM(in_size=5, hidden_size=hidden_size)
        self.mu_head = L.Linear(in_size=hidden_size, out_size=d_z)
        self.sigma_head = L.Linear(in_size=hidden_size, out_size=d_z)

        self.hidden_size = hidden_size
        self.d_z = d_z

    def forward(self, x):
        # hidden = self.lstm(x)

        # hidden = hidden[:,-1,:]        

        seq_len = x.shape[1]
        h, c = None, None
        for i in range(seq_len):
            if h is None or c is None:
                h, c = self.lstm(x[:, i, :])
            else:
                h, c = self.lstm(x[:, i, :], h, c)

        mu = self.mu_head(h)
        sigma_hat = self.sigma_head(h)
        sigma = F.exp(sigma_hat / 2.)

        xp = cuda.get_array_module(mu)
        z = mu + sigma * xp.random.normal(0, 1, mu.shape)
        return z, mu, sigma
    



class Decoder(Model):
    def __init__(self, d_z, hidden_size, n_distributions):
        super().__init__()
        self.lstm = L.LSTM(in_size=d_z+5, hidden_size=hidden_size)

        self.init_h = L.Linear(in_size=d_z, out_size=hidden_size)
        self.init_c = L.Linear(in_size=d_z, out_size=hidden_size)

        self.pi_head = L.Linear(in_size=hidden_size, out_size=n_distributions)
        self.mu_x_head = L.Linear(in_size=hidden_size, out_size=n_distributions)
        self.mu_y_head = L.Linear(in_size=hidden_size, out_size=n_distributions)
        self.sigma_x_head = L.Linear(in_size=hidden_size, out_size=n_distributions)
        self.sigma_y_head = L.Linear(in_size=hidden_size, out_size=n_distributions)
        self.rho_xy_head = L.Linear(in_size=hidden_size, out_size=n_distributions)

        self.q_head = L.Linear(in_size=hidden_size, out_size=3)

        self.n_distributions = n_distributions
        self.hidden_size = hidden_size

    def forward(self, x, z, h=None, c=None):
        xp = cuda.get_array_module(x)
        h, c = None, None
        if h is None and c is None:
            h = F.tanh(self.init_h(z))
            c = F.tanh(self.init_c(z))

        seq_len = x.shape[1]
        
        outputs = None
        for i in range(seq_len):
            h, c = self.lstm(x[:, i, :], h, c)
            if outputs == None:
                outputs = F.expand_dims(h, 1)
            else:
                outputs = F.cat((outputs, F.expand_dims(h, 1)), axis=1)

        # hidden Needs to chagned to output of lstm
        # print(outputs.shape)
        # print(self.q_head(outputs).shape)

        outputs= F.reshape(outputs, (-1, self.hidden_size))
        q_logits = F.log_softmax(self.q_head(outputs))
        # print(q_logits.shape, "q_logits")

        pi_logits = self.pi_head(outputs)
        mu_x = self.mu_x_head(outputs)
        mu_y = self.mu_y_head(outputs)
        sigma_x = self.sigma_x_head(outputs)
        sigma_y = self.sigma_y_head(outputs)
        rho_xy = self.rho_xy_head(outputs)

        pi_logits = F.reshape(pi_logits, (-1, seq_len, self.n_distributions))
        mu_x = F.reshape(mu_x, (-1, seq_len, self.n_distributions))
        mu_y = F.reshape(mu_y, (-1, seq_len, self.n_distributions))
        sigma_x = F.reshape(sigma_x, (-1, seq_len, self.n_distributions))
        sigma_y = F.reshape(sigma_y, (-1, seq_len, self.n_distributions))
        rho_xy = F.reshape(rho_xy, (-1, seq_len, self.n_distributions))
        
        q_logits = F.reshape(q_logits, (-1, seq_len, 3))


        bgm = BivariateGaussianMixture(pi_logits, mu_x, mu_y, F.exp(sigma_x), F.exp(sigma_y), F.tanh(rho_xy))
        return bgm, q_logits, h, c


def ReconstructionLoss(mask, target, bgm, q_logits):
        xp = cuda.get_array_module(mask)
        # target is a 3 dimensional array
        # xy = target[:, :, 0:2].unsqueeze(-2).expand(-1, -1, dist.n_distributions, -1)
        xy = target[:, :, 0:2]
        x = xy[:, :, 0]
        y = xy[:, :, 1]
        
        distributions = bgm.n_distributions
        stacked_x = None
        stacked_y = None
        for i in range(distributions):
            if stacked_x is None:
                stacked_x = F.expand_dims(x, axis=2)
                stacked_y = F.expand_dims(y, axis=2)
            else:
                stacked_x = F.cat((stacked_x, F.expand_dims(x, axis=2)), axis=2)
                stacked_y = F.cat((stacked_y, F.expand_dims(y, axis=2)), axis=2)
        
        loss_stroke = bgm.get_lossfunc(stacked_x, stacked_y, mask)
        
        loss_pen = -F.mean(F.mul(target[:,:,2:], q_logits))
        
        return F.add(loss_stroke, loss_pen)
        
        

def KLDivergenceLoss(mu, sigma):
    xp = cuda.get_array_module(mu)
    tmp = xp.ones(sigma.shape)
    inner_1 = F.add(tmp, sigma)
    inner_2 = F.add(F.pow(mu, 2), F.exp(sigma))
    inner = F.sub(inner_1, inner_2)
    tmp2 = xp.ones(inner.shape) * -2
    return F.mean(F.div(inner, tmp2))
        

class VAE(Model):
    def __init__(self, d_z, enc_hidden_size, dec_hidden_size, n_distributions):
        super().__init__()
        self.encoder = Encoder(d_z, enc_hidden_size)
        self.decoder = Decoder(d_z, dec_hidden_size, n_distributions)
        
        if use_gpu:
            self.encoder.to_gpu()
            self.decoder.to_gpu()

    def forward(self, x, t):
        z, mu, sigma = self.encoder(x)
        
        seq_len = x.shape[1]
        xp = cuda.get_array_module(z)

        expanded_z = F.expand_dims(z, axis=1)
        
        z_stack = None
        
        for i in range(seq_len - 1):
            if i == 0:
                z_stack = expanded_z
            else:
                z_stack = F.cat((z_stack, expanded_z), axis=1)
        # x = F.expand_dims(z, axis=1)

        inputs = F.cat((x[:,:-1], z_stack), axis=2)
        # inputs = dezero.as_variable(inputs)
        bgm, q_logits, _, _ = self.decoder(inputs, z)

        kl_loss = KLDivergenceLoss(mu, sigma)


        rec_loss = ReconstructionLoss(t, x[:,1:], bgm, q_logits)
        # 
        return kl_loss + rec_loss

d_z = 4
enc_hidden_size = 32
dec_hidden_size = 64
n_distributions = 8
epochs = 10
batch_size = 100

from dezero.optimizers import Adam

data = np.load('./data/sketchrnn_apple.npz', encoding='latin1', allow_pickle=True)


strokes = StrokesDataset(data['train'], batch_size=batch_size, max_seq_length=200, gpu=False, shuffle=False)

model = VAE(d_z, enc_hidden_size, dec_hidden_size, n_distributions)

if use_gpu:
    strokes.to_gpu()
    model.to_gpu()

optimizer = Adam().setup(model)

for epoch in range(epochs):
    loss = 0
    for i in range(strokes.max_iter):
        x, t = strokes.__next__()
        loss = model(x, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        # print(loss.data)
        loss = loss.data
        print("working?", loss)
        
    print(f"Epoch {epoch}, loss: {loss}")

model.save_weights("sketch_rnn.npz")

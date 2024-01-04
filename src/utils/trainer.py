import torch
import numpy as np
import abc

mae_crit = torch.nn.L1Loss()

class BaseTrainer:
    def train(self, dataloader):
        train_loss = 0
        self.model.train()
        for batch in dataloader:
#        for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc='train'):
            self.opt.zero_grad()
            out = self._iter_step(batch)

            loss = mae_crit(out[0], out[1])
            loss.backward()
            self.opt.step()
            
            train_loss += loss.item()
        return train_loss / len(dataloader)
    
    def test(self, dataloader):
        test_loss = 0
        outputs   = {}
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
#            for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc='test'):
                out = self._iter_step(batch)    
                test_loss += mae_crit(out[0], out[1]).item()
                outputs = self._get_outputs(batch, out, outputs)
        outputs = self._parse_outputs(outputs)
        return test_loss / len(dataloader), outputs
    
    def pred(self, dataloader):
        outputs = {}
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
            #for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc='pred'):
                out = self._iter_step(batch)
                outputs = self._get_outputs(batch, out, outputs)
        outputs = self._parse_outputs(outputs)
        return outputs
    
    @abc.abstractmethod
    def _iter_step(self, batch):
        pass

    @abc.abstractmethod
    def _get_outputs(self, batch, output, outputs):
        pass

    @abc.abstractmethod
    def _parse_outputs(self, outputs):
        pass

class DBPNTrainer(BaseTrainer):
    def __init__(self, model, opt, residual=True, device=torch.device('cuda')):
        self.model = model
        self.opt   = opt
        self.residual = residual
        self.device = device

    def _iter_step(self, batch):
        inp, tgt, bic, _ = batch
        inp = inp.to(self.device)
        tgt = tgt.to(self.device)
        bic = bic.to(self.device)
        
        pred = self.model(inp)
        if self.residual:
            pred += bic
        return pred, tgt

    def _get_outputs(self, batch, output, outputs):
        _, tgt, bic, info = batch
        pred_img = output[0].cpu().clamp(0,1).numpy()
        tgt_img  = tgt.cpu().clamp(0,1).numpy()
        bic_img  = bic.cpu().clamp(0,1).numpy()
        for i, t, p, b in zip(info, tgt_img, pred_img, bic_img):
            tag = f'{i[0]}_{i[1]}'
            if tag not in outputs.keys():
                outputs[tag] = {
                    'vmin':float(i[2]), 'vmax':float(i[3]),
                    'x':[], 'pred':[], 'bic':[], 'tgt':[]
                }
            outputs[tag]['x'].append(float(i[4]))
            outputs[tag]['bic'].append(b)
            outputs[tag]['tgt'].append(t)
            outputs[tag]['pred'].append(p)
        return outputs
    
    def _parse_outputs(self, outputs):
        for v in outputs.values():
            o = np.argsort(v['x'])
            v['x'] = np.array(v['x'])[o].squeeze()
            v['tgt'] = np.stack(v['tgt'], axis=0)[o].squeeze()
            v['pred'] = np.stack(v['pred'], axis=0)[o].squeeze()
            v['bic'] = np.stack(v['bic'], axis=0)[o].squeeze()
        return outputs

class SwinIRTrainer(BaseTrainer):
    def __init__(self, model, opt, device=torch.device('cuda')):
        self.model  = model
        self.opt    = opt
        self.device = device
        
    def _iter_step(self, batch):
        inp, tgt, _, _ = batch
        inp = inp.to(self.device)
        tgt = tgt.to(self.device)
        
        pred = self.model(inp)[:, :1]
        return pred, tgt
    
    def _get_outputs(self, batch, output, outputs):
        _, tgt, bic, info = batch
        pred_img = output[0].cpu().clamp(0,1).numpy()
        tgt_img  = tgt.cpu().clamp(0,1).numpy()
        bic_img  = bic.cpu().clamp(0,1).numpy()
        for i, t, p, b in zip(info, tgt_img, pred_img, bic_img):
            tag = f'{i[0]}_{i[1]}'
            if tag not in outputs.keys():
                outputs[tag] = {
                    'vmin':float(i[2]), 'vmax':float(i[3]),
                    'x':[], 'pred':[], 'bic':[], 'tgt':[]
                }
            outputs[tag]['x'].append(float(i[4]))
            outputs[tag]['bic'].append(b[:, :i[5], :i[6]])
            outputs[tag]['tgt'].append(t[:, :i[5], :i[6]])
            outputs[tag]['pred'].append(p[:, :i[5], :i[6]])
        return outputs
    
    def _parse_outputs(self, outputs):
        for v in outputs.values():
            o = np.argsort(v['x'])
            v['x'] = np.array(v['x'])[o].squeeze()
            v['tgt'] = np.stack(v['tgt'], axis=0)[o].squeeze()
            v['pred'] = np.stack(v['pred'], axis=0)[o].squeeze()
            v['bic'] = np.stack(v['bic'], axis=0)[o].squeeze()
        return outputs


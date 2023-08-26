# coding=utf-8
import torch
import torch.nn.functional as F

# Process-based Contrastive Loss (PCL)
class PCL(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_segments = cfg.TRAIN.NUM_SEGMENTS
        self.temperature = cfg.OPTIMIZER.TEMPERATURE

    def vsp_loss(self, model, videos, bridges, labels, names):
        """Two types inputs for the model.
        Single segment: penn_action,finegym
        Multiple segments: pouring, kinetics400
        """
        num_frames = self.cfg.TRAIN.NUM_FRAMES
        embs = model(videos, num_frames, video_masks=None)
        batch_size, num_seg, _ = bridges.shape
        points = torch.empty(batch_size,num_seg)
        for i in range(batch_size):
            for j in range(num_seg):
                points[i][j] = bridges[i][j][1]

        pcl = self.process_contrastive_loss(embs, bridges)

        if self.cfg.DATA.ANNOTATION == 'frame':
            scl = self.supervised_contrastive_loss(embs, points, labels, names)
            return scl + pcl
        else:
            return pcl

    def process_contrastive_loss(self, embs, bridges):
        batch_size, num_seg, _ = bridges.shape
        loss = 0
        for i in range(batch_size):
            cur_video_emb = embs[i]
            cur_video_bridges = bridges[i]
            cur_video_loss = 0
            for j in range(num_seg):
                cur_bridge = cur_video_bridges[j]
                cur_bridge_index = num_seg * i + j
                cur_bridge_emb_head = cur_video_emb[cur_bridge[0].long()]
                cur_bridge_emb_tail = cur_video_emb[cur_bridge[2].long()]
                cur_bridge_emb = [cur_bridge_emb_head,cur_video_emb[cur_bridge[1].long()],cur_bridge_emb_tail]
                pos_dis = self.Brownian_bridge_distance(cur_bridge_emb, cur_bridge)
                numer = torch.exp(pos_dis)
                deno = self.neg_sum(embs,bridges,cur_bridge,cur_bridge_index,cur_bridge_emb_head,cur_bridge_emb_tail)
                #cur_bridge_loss
                cur_video_loss += -torch.log(numer/(numer+deno))
            cur_video_loss /= num_seg
            loss += cur_video_loss
        return loss / batch_size

    def supervised_contrastive_loss(self,embs,points,labels,names):
        batch_size, num_seg = points.shape
        loss = 0
        if num_seg == 1:
            for i in range(batch_size):
                numer,deno = 0,0
                for j in range(i,batch_size):
                    similarity = torch.sum(torch.mul(embs[i][points[i][0].long()], embs[j][points[j][0].long()]))
                    if names[i][5:] == names[j][5:] and labels[i][0] == labels[j][0]:
                        numer += torch.exp(torch.true_divide(similarity,self.temperature))  
                    else:
                        deno += torch.exp(torch.true_divide(similarity,self.temperature))  
                loss += torch.true_divide(numer,numer+deno) 
        else:
            for i in range(batch_size):
                cur_video_loss = 0
                for j in range(i,batch_size):
                    cur_video_loss += self.matrix_loss(embs[i],embs[j],points[i],points[j])
                loss += cur_video_loss
        return loss / batch_size

    def matrix_loss(self,va,vb,va_p,vb_p):
        vta = [va[i.long()] for i in va_p]
        vtb = [vb[i.long()] for i in vb_p]
        A = torch.stack(vta)
        B = torch.stack(vtb)
        logits_vta = A @ B.T
        labels = torch.arange(A.shape[0], dtype=torch.long)
        loss = F.cross_entropy(logits_vta, labels)
        return loss


    def Brownian_bridge_distance(self, emb, bridge):
        bh,bp,bt = bridge[0],bridge[1],bridge[2]
        alpha = torch.true_divide(bp-bh,bt-bh) 
        sigma = alpha * (bt-bp)
        x = emb[1] - (1-alpha)*emb[0] - alpha*emb[2]

        return -torch.norm(x,p=2)**2 / (2*sigma**2)

    def neg_sum(self,embs,bridges,cur_bridge,cur_bridge_index,cur_head,cur_tail):
        batch_size, num_seg, _ = bridges.shape #!!!
        deno = 0
        for i in range(batch_size):
            cur_neg_emb = embs[i]
            cur_neg_bridges = bridges[i]
            for j in range(num_seg):
                cur_neg_bridge = cur_neg_bridges[j]
                if num_seg * i + j != cur_bridge_index:
                    for point in cur_neg_bridge:
                        neg_bridge_emb = [cur_head,cur_neg_emb[point.long()],cur_tail]
                        neg_dis = self.Brownian_bridge_distance(neg_bridge_emb, cur_bridge)
                        deno += torch.exp(neg_dis)
        return deno

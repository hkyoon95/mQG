import torch
import torch.nn.functional as F

def mqs_loss(score_matrix, sent_num=0, prefix_len=0):
    bsz, seqlen, _ = score_matrix.size()
    gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2) # bsz x seqlen
    gold_score = torch.unsqueeze(gold_score, -1)
    assert gold_score.size() == torch.Size([bsz, seqlen, 1])
    difference_matrix = gold_score - score_matrix # lower similar
    assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])
    loss_matrix = 0.5 * difference_matrix 
    loss_matrix = torch.nn.functional.relu(loss_matrix)
    input_mask = []
    for i in range(bsz):
        temp = torch.tensor([1]*(sent_num[i]+1)).type(torch.FloatTensor)
        temp = F.pad(temp, (0,max(sent_num)-sent_num[i]), value=0.0)
        input_mask.append(temp)
    input_mask = torch.stack(input_mask)
    if loss_matrix.is_cuda:
        input_mask = input_mask.cuda(loss_matrix.get_device())
    # for each batch calculate valid_len_list
    valid_len_list = torch.sum(input_mask, dim = -1).tolist() # bsz

    # for each batch calculate valid_len_list
    valid_len_list = torch.sum(input_mask, dim = -1).tolist()
    loss_mask = build_mask_matrix_unsim(seqlen, [int(item) for item in valid_len_list], sent_num, prefix_len)
    if score_matrix.is_cuda:
        loss_mask = loss_mask.cuda(score_matrix.get_device())
    masked_loss_matrix = loss_matrix * loss_mask

    loss_matrix = torch.sum(masked_loss_matrix, dim = -1)
    # assert loss_matrix.size() == input_ids.size()
    loss_matrix = loss_matrix * input_mask
    sent_loss = torch.sum(loss_matrix) / torch.sum(loss_mask)
    
    return sent_loss

def build_mask_matrix_unsim(seqlen, valid_len_list, sent_num, prefix_len=0):

    res_list = []
    base_mask = torch.ones(seqlen, seqlen) - torch.eye(seqlen, seqlen)
    base_mask = base_mask.type(torch.FloatTensor)
    bsz = len(valid_len_list)

    # make mask for each batch
    # mask for pad seqs
    for i in range(bsz):
        one_base_mask = base_mask.clone()
        one_valid_len = valid_len_list[i]
        one_valid_sent_num = sent_num[i]

        if one_valid_sent_num!=0:
            one_base_mask[:,one_valid_len-one_valid_sent_num:-one_valid_sent_num] = 0.
            one_base_mask[one_valid_len-one_valid_sent_num:-one_valid_sent_num, :] = 0.       
        else:
            one_base_mask[:,one_valid_len:] = 0.
            one_base_mask[one_valid_len:, :] = 0.
        one_base_mask[1:, 1:] = 0.
        if prefix_len > 0:
            one_base_mask[:prefix_len, :prefix_len] = 0.
        res_list.append(one_base_mask)
    res_mask = torch.stack(res_list, dim = 0) # torch.FloatTensor(res_list)
    assert res_mask.size() == torch.Size([bsz, seqlen, seqlen])
    return res_mask
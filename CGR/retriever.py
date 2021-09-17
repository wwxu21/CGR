from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
import torch
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.nn import functional as F

class Retriever(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert= BertModel(config)
        self.init_weights()
    def forward(
        self,
        query_ids=None,
        query_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        fact_vec=None,
        iter_mask=None,
        return_dict=None,
        label=None,
    ):
        outputs_hypo = self.bert(
            query_ids,
            attention_mask=query_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if fact_vec is None:
            query_state = outputs_hypo['last_hidden_state'][:, 0, :]
            query_state = F.normalize(query_state, p=2, dim=1)
            return query_state
        else:
            query_state = outputs_hypo['last_hidden_state'][:,0,:]
            query_state = F.normalize(query_state, p=2, dim=1)
            fact_state = fact_vec
            batch_size = label.size(0)
            iter_size = iter_mask.size(0)
            fact_size = fact_state.size(0)

            hypo_state = query_state[:batch_size]
            chain_iter_query_state = query_state[batch_size: batch_size + iter_size]
            gold2_query_state = query_state[batch_size + iter_size:]

            chain_global_state = fact_state[fact_size - iter_size - batch_size:fact_size - iter_size]
            chain_iter_fact_state = fact_state[fact_size - iter_size:]
            gold1_state = fact_state[max(0, fact_size - iter_size - 2 * batch_size):max(0, fact_size - iter_size - batch_size)]
            gold2_state = fact_state[max(0, fact_size - iter_size - 3 * batch_size):max(0, fact_size - iter_size - 2 * batch_size)]

            loss_fct = CrossEntropyLoss()
            loss = 0
            dot_sim_iter = None
            if gold1_state.size()[0] != 0:
                gold1_label = torch.arange(0, gold1_state.size()[0]).type_as(query_ids)
                dot_sim_gold1 = torch.mm(hypo_state, gold1_state.t())
                loss_gold1 = loss_fct(dot_sim_gold1, gold1_label)
                loss += loss_gold1
            if gold2_state.size()[0] != 0:
                gold2_label = torch.arange(0, gold2_state.size()[0]).type_as(query_ids)
                dot_sim_gold2 = torch.mm(gold2_query_state, gold2_state.t())
                loss_gold2 = loss_fct(dot_sim_gold2, gold2_label)
                loss += loss_gold2
            if chain_global_state.size()[0] != 0:
                global_label = torch.arange(0, chain_global_state.size()[0]).type_as(query_ids)
                dot_sim_global = torch.mm(hypo_state, chain_global_state.t())
                loss_global = loss_fct(dot_sim_global, global_label)
                loss += loss_global
            if chain_iter_query_state.size()[0] != 0:
                iter_label = torch.arange(0, chain_iter_query_state.size()[0]).type_as(query_ids)
                dot_sim_iter = torch.mm(chain_iter_query_state, chain_iter_fact_state.t())
                dot_sim_iter.masked_fill_(iter_mask, -10000)
                loss_iter = loss_fct(dot_sim_iter, iter_label)
                loss += loss_iter

            outputs = (loss,) + (dot_sim_iter,)
            return outputs

    def RLlearning(self,
                   policy=None,
                   pred=None,
                   label=None,
                   iter_segmentation=None):
        if policy is None:
            return 0
        batch_size = label.size(0)
        policy = torch.softmax(policy, dim=1)
        policy = -torch.log(policy[range(iter_segmentation[-1]),range(iter_segmentation[-1])])
        pred = torch.argmax(pred, dim=1)
        RL_loss = 0
        reward_bias = 0
        policy_bias = 0
        t = 0
        for ib in range(batch_size):
            if pred[ib] == label[ib]:
                reward = 1 - reward_bias
            else:
                reward = 0 - reward_bias
            if ib == 0:
                start = 0
            else:
                start = iter_segmentation[ib - 1]
            end = iter_segmentation[ib]
            if start != end:
                t += 1
                reward_bias += reward
                policy_one = torch.sum(policy[start: end])/ (end - start)
                policy_bias += policy_one
                RL_loss += reward * policy_one
        RL_loss = (RL_loss - reward_bias * policy_bias / t) / t
        return RL_loss

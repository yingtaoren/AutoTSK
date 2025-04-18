import math
import torch
import torch.nn as nn

from utils.util import reset_params


class AutoTSK(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        n_rule,
        antecedent,
        order=1,
        eps=1e-8,
        precons=None,
        gate_to_conseq=False,
        attention_to_conseq=False,
    ):
        super(AutoTSK, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_rule = n_rule
        self.antecedent = antecedent
        self.precons = precons
        self.device = "cuda"

        self.order = order
        assert self.order == 0 or self.order == 1, "Order can only be 0 or 1."
        self.eps = eps
        self.rule_layers = None
        self.gate_to_conseq = gate_to_conseq
        self.attention_to_conseq = attention_to_conseq

        self.__build_model__()

    def set_gate_to_conseq(self):

        self.gate_to_conseq = True
        self.attention_to_conseq = False


    def set_attention_to_conseq(self):

        self.attention_to_conseq = True
        self.gate_to_conseq = False

    def set_raw_conseq(self): 

        self.gate_to_conseq = False
        self.attention_to_conseq = False


    def __build_model__(self):
        if self.order == 0:
            self.cons = nn.Linear(
                self.n_rule, self.out_dim, bias=True, dtype=torch.float32
            )
        else:
            self.rule_layers = nn.ModuleList(
                [nn.Linear(self.in_dim, self.out_dim) for _ in range(self.n_rule)]
            )

    def reset_parameters(self):

        reset_params(self.antecedent)

        if self.precons is not None:
            self.precons.reset_parameters()

    def forward(self, X, get_frs=False, get_rules_result=False):
        frs = self.antecedent(X)
        final_output = None

        if self.precons is not None:
            X = self.precons(X)

        if self.order == 0:
            # useless
            cons_input = frs
        else:
            if self.gate_to_conseq:
                outputs = []
                for i, layer in enumerate(self.rule_layers):

                    revised_X = X * (self.antecedent[0].alpha_gate[:, i] )

                    rule_output = layer(revised_X)  
                    rule_output_softmax = torch.nn.functional.softmax(
                        rule_output, dim=1
                    )  
                    outputs.append(
                        rule_output_softmax  * frs[:, i].unsqueeze(1)
                    ) 
                final_output = torch.sum(torch.stack(outputs, dim=-1), dim=-1)
                final_output = final_output.type(torch.float32)

            elif self.attention_to_conseq:
                outputs = []
                for i, layer in enumerate(self.rule_layers):

                    revised_X = X * (self.antecedent[0].alpha[:, i])

                    rule_output = layer(revised_X)
                    rule_output_softmax = torch.nn.functional.softmax(
                        rule_output, dim=1
                    )
                    outputs.append(
                        rule_output_softmax * frs[:, i].unsqueeze(1)
                    )               
                final_output = torch.sum(torch.stack(outputs, dim=-1), dim=-1)
                final_output = final_output.type(torch.float32)
            elif not self.gate_to_conseq and not self.attention_to_conseq:
                outputs = []
                for i, layer in enumerate(self.rule_layers):

                    X = X 

                    rule_output = layer(X)

                    rule_output_softmax = torch.nn.functional.softmax(
                        rule_output, dim=1
                    )  
                    outputs.append(rule_output_softmax * frs[:, i].unsqueeze(1))
                final_output = torch.sum(torch.stack(outputs, dim=-1), dim=-1)
                final_output = final_output.type(torch.float32)
            else:
                raise ValueError("Invalid consequent setting.")

        if get_frs:
            if get_rules_result:
                return final_output, frs, outputs
            return final_output, frs
        return final_output

import transformers
from_pretrained="facebook/wav2vec2-base" # "facebook/wav2vec2-base-960h",
# this key could be find in https://huggingface.co/models?sort=downloads&search=wav2vec
from_pretrained_num_hidden_layers=3
# how many layers in encoder you want to use if you don't use all encoder layer
pretrain_model = transformers.Wav2Vec2Model.from_pretrained(
	from_pretrained,
)

# optional code
pconfig = pretrain_model.config.to_dict()
num_hidden_layers = pconfig["num_hidden_layers"]
if from_pretrained_num_hidden_layers is not None:
    assert from_pretrained_num_hidden_layers <= num_hidden_layers, \
        f"from_pretrained_num_hidden_layers should equal or \
        samll than num_hidden_layers {num_hidden_layers} in config"
    pretrain_model.encoder.layers = \
        pretrain_model.encoder.layers[:from_pretrained_num_hidden_layers]
    pretrain_model.config.num_hidden_layers = from_pretrained_num_hidden_layers
print(from_pretrained)


# call

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.
    See description of make_non_pad_mask.
    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.
    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask
xs = torch.rand([5, 32, 80])
xs_lens = torch.tensor([14, 15, 32])
masks = ~make_pad_mask(xs_lens)
output = from_pretrained(
	input_values = xs,
	attention_mask = masks,
)
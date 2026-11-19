import torch
from peft import PeftModel
from transformers import WhisperConfig, WhisperForConditionalGeneration

from speechlab.asr import attach_lora, collate_cached


def test_whisper_audio_forward_adapter_training_and_round_trip(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(13)
    config = WhisperConfig(vocab_size=40, num_mel_bins=8, d_model=16, encoder_layers=1, decoder_layers=1,
                           encoder_attention_heads=2, decoder_attention_heads=2,
                           encoder_ffn_dim=32, decoder_ffn_dim=32, max_source_positions=10,
                           max_target_positions=16, decoder_start_token_id=1, bos_token_id=1,
                           eos_token_id=2, pad_token_id=0, suppress_tokens=[], begin_suppress_tokens=[])
    model = attach_lora(WhisperForConditionalGeneration(config), rank=2)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    adapters = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    features = torch.randn(2, 8, 20)
    labels = torch.tensor([[5, 6, 2], [7, 2, -100]])
    # This catches the PEFT text-wrapper/input_features incompatibility.
    loss = model(input_features=features, labels=labels).loss
    assert torch.isfinite(loss)
    loss.backward()
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01).step()
    assert any(not torch.equal(adapters[n], p) for n, p in model.named_parameters() if p.requires_grad)
    assert all(torch.equal(frozen[n], p) for n, p in model.named_parameters() if not p.requires_grad)
    model.eval()
    with torch.no_grad():
        expected = model(input_features=features, labels=labels).logits
        hidden = model.get_base_model().get_encoder()(features).last_hidden_state
        batch = collate_cached([{"hidden": hidden[0], "labels": [5, 6, 2]}, {"hidden": hidden[1], "labels": [7, 2]}], "cpu")
        assert batch["labels"][1, 2].item() == -100
        cached = model(**batch).logits
        torch.testing.assert_close(expected, cached)
    model.save_pretrained(tmp_path / "adapter")
    reloaded = PeftModel.from_pretrained(model.unload(), tmp_path / "adapter").eval()
    with torch.no_grad():
        torch.testing.assert_close(expected, reloaded(input_features=features, labels=labels).logits)

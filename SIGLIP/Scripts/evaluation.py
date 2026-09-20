"""Training/evaluation pipeline - port of the original MIMOSA Scripts/evaluation.py."""
import SIGLIP.Scripts.models as m


def pipline(
    train_loader,
    valid_loader,
    test_loader,
    model_path,
    n_heads,
    epochs,
    lr_rate,
    num_classes=4,
    seq_len=70,
    attn_variant="code",
    fix_scheduler=False,
    checkpoint_name="maf_model.pth",
):

    # call the train function
    model = m.train(
        train_loader,
        valid_loader,
        model_path,
        n_heads,
        epochs,
        lr_rate,
        num_classes=num_classes,
        seq_len=seq_len,
        attn_variant=attn_variant,
        fix_scheduler=fix_scheduler,
        checkpoint_name=checkpoint_name,
    )

    # call test function
    actual_labels, pred_labels = m.evaluation(model_path, model, test_loader, checkpoint_name=checkpoint_name)

    return actual_labels, pred_labels

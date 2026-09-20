import models as m

def pipline(train_loader, valid_loader, test_loader, model_path, n_heads, epochs, lr_rate):

  # call the train function 
  model = m.train(train_loader, valid_loader, model_path, n_heads, epochs, lr_rate)

  # call test function
  actual_labels, pred_lables = m.evaluation(model_path, model, test_loader)

  return actual_labels, pred_lables
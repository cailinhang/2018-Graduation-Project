import torch
from torch.autograd import Variable


def to_var(x, requires_grad=False, volatile=False):
    
#    自动选择 cpu 或 cuda
    
#    if torch.cuda.is_available():
#        x = x.cuda()
        
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

    
def train(model, loss_fn, optimizer, param, loader_train, loader_val=None):

    model.train()
    for epoch in range(param['num_epochs']):
        #print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

        for t, (x, y) in enumerate(loader_train):
            #x_var, y_var = to_var(x), to_var(y.long())
            x_var, y_var = Variable(x), Variable(y)                            
            scores = model(x_var)
            loss = loss_fn(scores, y_var)

#            if (t + 1) % 100 == 0:
#                print('t = %d, loss = %.8f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
         

def test(model, loader):

    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)
    
    for x, y in loader:
        #x_var = to_var(x, volatile=True)
        with torch.no_grad():
                x_var = Variable(x)
                
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / num_samples

    print('Test accuracy: {:.2f}% ({}/{})'.format(
        100.*acc,
        num_correct,
        num_samples,
        ))
    
    return acc
    

    

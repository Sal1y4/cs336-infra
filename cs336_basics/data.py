import torch
def get_batch(dataset, batch_size, context_length, device):
    starts = torch.randint(
        low=0,
        high=len(dataset) - context_length,
        size=(batch_size,),
    )
    

    x_list = []
    y_list = []

    for start in starts:
        start = start.item()

        x_row = dataset[start : start + context_length]
        y_row = dataset[start + 1 : start + context_length + 1]

        x_list.append(torch.tensor(x_row, dtype=torch.long))
        y_list.append(torch.tensor(y_row, dtype=torch.long))

    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)

    return x, y
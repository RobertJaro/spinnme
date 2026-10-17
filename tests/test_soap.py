import torch

from spinnme.train.soap import SOAP


def test_closure_runs_with_gradients_enabled():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = SOAP([parameter], lr=0.1)
    closure_state = {}

    def closure():
        closure_state['grad_enabled'] = torch.is_grad_enabled()
        loss = parameter.square()
        closure_state['requires_grad'] = loss.requires_grad
        loss.backward()
        return loss

    loss = optimizer.step(closure)

    assert closure_state == {'grad_enabled': True, 'requires_grad': True}
    assert loss.requires_grad
    assert parameter.grad is not None


def test_parameter_updates_after_preconditioner_initialization():
    parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    optimizer = SOAP([parameter], lr=0.1)

    def step():
        optimizer.zero_grad()

        def closure():
            loss = parameter.square().sum()
            loss.backward()
            return loss

        return optimizer.step(closure)

    initial = parameter.detach().clone()
    first_loss = step()
    after_initialization = parameter.detach().clone()
    second_loss = step()

    # SOAP deliberately uses the first gradient only to initialize its
    # preconditioner, then starts changing parameters on the next step.
    torch.testing.assert_close(after_initialization, initial)
    assert not torch.equal(parameter.detach(), initial)
    assert second_loss.item() == first_loss.item()

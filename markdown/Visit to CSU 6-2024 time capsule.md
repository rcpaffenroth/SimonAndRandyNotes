# Week 1

Results from first week are contained in

Training paper
[0 Start here](Training%20paper/0%20Start%20here.md)

which is also in 

[Training paper with Simon et al - Online LaTeX Editor Overleaf](https://www.overleaf.com/project/665485d7b41871560fdb91ae)

Code is in

[Release csu_simon_results · rcpaffenroth/inn_survey (github.com)](github.com))

and wandb plots are in

[inn_survey_KEEP_FOR_SIMON_6-19 Workspace – Weights & Biases (wandb.ai)](wandb.ai))

# 6-13-2024

This does not seem to converge
```bash 
python demo_2_dynamcial_systems.py --data_name=MNIST1D --model_type=I_smart_three_by_three --loss=bce
```
but this does
```bash 
python demo_2_dynamcial_systems.py --data_name=MNIST1D --model_type=I_smart_three_by_three --loss=mse
```
But the I_MLP version seems to work ok with bce, need to look into.

Note, I can use wandb parameter sweeps even though the code it note setup for wandb plotting (which will be the next version)

Need to refactor get_model, keep state.x, state.y, and state.z separate?  Also h_size is kludgy
## Notes


https://stackoverflow.com/questions/66272911/how-to-see-the-data-in-dataloader-in-pytorch

F may want to see all the data at once, including knowing what the batches will be, so it can cache for efficiency 

Why is $i$ not internal to F?  A normal nn has exactly that property.  In our case we want to have a loss on the intermediate steps!  That is really a key point and why the code is different.  We want insights as to the thinking.  As little "hidden" as possible...


https://stackoverflow.com/questions/72635102/what-is-the-azure-equivalent-of-gcps-cloud-run

https://huggingface.co/docs/hub/spaces

Gradio and streamlit

# 6-14-2024

he following comment is very important in inn_survey

```python
        # Note, this is not necessarily equivalent to the LS solution above!
        # The above accounts for information in the y part of start being useful
        # for computing the y part of target.  This model does not account for that.
        # More perversely, this may not even be as good as the identity function!
        # For the the y part of start is the same as the y part of target, or the y part
        # of start is a good guess and the x part of start is not informative for the y part
        # of target!
```

# 6-15-2024

- [ ] Get cleaner setup for distributed runs
	- [ ] Make sure all machines have most recent git checkout
		- [ ] iterativennsimple
		- [ ] inn_survey
	- [ ] Make sure all machines have a recent "poetry install" 
	- [ ] Perhaps ansible?
		- [ ] Can also run script
		- [ ] How to handle GPUs?  Perhaps different hosts in inventory for different GPU numbers?
- [ ] Learn about wandb reports
- [ ] *Do initial runs in inn_sequence*
	- [ ] This one will take a lot of thought, perhaps do early Sunday morning when I am fresh?
- [ ] Think about making things faster on GPU

# 6-16-2024
Ok, I am doing something wrong.  Assume I have

$$
z_0=
\begin{bmatrix}
x_0 \\
y_0 
\end{bmatrix}
$$
and
$$
z_1=
\begin{bmatrix}
x_1 \\
y_1 
\end{bmatrix}
$$
Note that the split into $x$ and $y$ revolves around the standard ML case where $x_0=x_1$, but $y_0 \ne y_1$.  In fact for the standard ML case, we ignore $y_0$ and write $y_1 \approx f(x_0;\theta)$.

Now, I have been mixing up $x_0$ and $z_0$, which play very different roles!  For example, I like to write

$$
\begin{bmatrix}
I & 0 \\
W & 0 
\end{bmatrix}
\cdot
\begin{bmatrix}
x_0 \\
y_0 
\end{bmatrix}=
\begin{bmatrix}
x_0 \\
W  \cdot x_0 
\end{bmatrix}\approx
\begin{bmatrix}
x_0 \\
y_1 
\end{bmatrix}
$$
when I want to predict $y_1$ from $x_0$ and I can write a very similar equation 
$$
\begin{bmatrix}
I & 0 \\
W & 0 
\end{bmatrix}
\cdot
\begin{bmatrix}
z_0 \\
z_1 
\end{bmatrix}=
\begin{bmatrix}
z_0 \\
W  \cdot z_0 
\end{bmatrix}\approx
\begin{bmatrix}
z_0 \\
z_1 
\end{bmatrix}
$$
when I want to predict $z_1$ from $z_0$.  Note, both of these maps have the nice property that they are idempotent, so if I iterate them I don't change the output.

However, unless I am careful (which I have not been) following this line of reasoning starts getting me into trouble when I do continuation!  Starting from
$$
W
\cdot
z_0
\approx
z_1
$$
I embed this into a larger problem in what might be called boosting, and write things like

$$
\begin{bmatrix}
W & c(\gamma_0)\equiv0 \\
f(\cdot;\theta_f) & g(\cdot;\theta_g)
\end{bmatrix}
\begin{bmatrix}
z_0 \\
h_0
\end{bmatrix}
=
\begin{bmatrix}
W \cdot  z_0 \\
f(z_0;\theta_f) + g(h_0;\theta_g)
\end{bmatrix}
\approx
\begin{bmatrix}
z_1 \\
f(z_0;\theta_f)+ g(h_0;\theta_g)
\end{bmatrix}
$$
and I get a nice correction term of the form
$$
\begin{bmatrix}
W & c(\gamma_1) \\
f(\cdot;\theta_f) & g(\cdot;\theta_g)
\end{bmatrix}
\begin{bmatrix}
z_0 \\
h_0
\end{bmatrix}
=
\begin{bmatrix}
W  \cdot z_0 + c(\gamma_1)\cdot h_0 \\
f(z_0;\theta_f) + g(h_0;\theta_g)
\end{bmatrix}
\approx
\begin{bmatrix}
z_1 \\
f(z_0;\theta_f)+ g(h_0;\theta_g)
\end{bmatrix}
$$

Awesome, $c(\gamma_1)\cdot h_0$ can be learned to help correct our LS approximation $W \cdot z_0$ 😎 ... 
except that $f$ and $g$ play no role in the approximation of $z_1$! :-( In particular, $h_0$ is some arbitrary initialization of $h$ and is independent of $z_0$! 🫤

Of course, we know how to fix this, we just iterate the map again

$$
\begin{bmatrix}
W & c(\gamma_1) \\
f(\cdot;\theta_f) & g(\cdot;\theta_g)
\end{bmatrix}
\begin{bmatrix}
W  \cdot z_0 + c(\gamma_1)\cdot h_0 \\
f(z_0;\theta_f) + g(h_0;\theta_g)
\end{bmatrix}
=
\begin{bmatrix}
W \cdot( W  \cdot z_0 + c(\gamma_1)\cdot h_0) + c(\gamma_1) \cdot (f(z_0;\theta_f) + g(h_0;\theta_g)) \\
\cdots
\end{bmatrix}
$$

Ok, we are now all set... The correction term $c(\gamma_1) \cdot (f(z_0;\theta_f) + g(h_0;\theta_g))$ fulfills all of our hopes and dreams.  It is non-linear if $f$ and/or $g$ are are non-linear, it depends on $z_0$, and it can be learned by optimizing $\gamma$ and $\theta_f$ (we don't actually care about $\theta_g$ yet since $g$ is independent of $z_0$, but one more iteration will fix that too 😎.

Except, we are correcting the wrong thing 🤯 

Let's look carefully at the term we are correcting, namely

$$
W \cdot( W \cdot z_0 + c(\gamma_1)\cdot h_0)
$$
expanding this reveals the error of our ways
$$
W \cdot W \cdot z_0 + W \cdot (c(\gamma_1)\cdot h_0)
$$
Unfortunately, the least squares solution $W \cdot z_0$ that we are hoping to correct is *no-where to be found* 😒
First, the LS matrix $W$ is not in any way optimal when used in the term $W \cdot (c(\gamma_1)\cdot h_0)$.  This is basically just some noise term.
Second, and even worse, while $W\cdot z_0$ is an optimal approximation of $z_1$, $W$ is *not idempotent*. So, $W \cdot (W \cdot z_0)$ has nothing to do with $z_1$!

We finally have a useful correction, but to the wrong thing 😖  In fact, more iterations just make things worse, since we just end up with higher and higher powers of $W \cdot \cdots \cdot W \cdot z_0$ and, depending on the eigenvalues of $W$, things can quickly diverge.

So, what to do? We need to be careful thinking about $z$ versus $x$.

$$
\begin{bmatrix}
I & 0 & 0 \\
W_z & 0 & c(\gamma_0)\\
F & f & g 
\end{bmatrix}
\cdot
\begin{bmatrix}
z_0 \\
z_0 \\
h_0
\end{bmatrix}
=
\begin{bmatrix}
z_0 \\
W_z \cdot z_0 + c(\gamma_0) \cdot h_0 \\
F \cdot z_0 + f(z_0) + g(h_0)
\end{bmatrix}
=
\begin{bmatrix}
z_0 \\
W_z \cdot z_0 \\
F \cdot z_0 + f(z_0) + g(h_0)
\end{bmatrix}\approx
\begin{bmatrix}
z_0 \\
z_1 \\
F \cdot z_0 + f(z_0) + g(h_0) 
\end{bmatrix}
$$

Now, when we iterate again we get what we want

$$
\begin{bmatrix}
I & 0 & 0 \\
W_z & 0 & c(\gamma_0)\\
F & f & g 
\end{bmatrix}
\cdot
\begin{bmatrix}
z_0 \\
W_z \cdot z_0 + c(\gamma_0) \cdot h_0 \\
H(z_0, h_0)
\end{bmatrix}
=
\begin{bmatrix}
z_0 \\
W_z \cdot z_0 + c(\gamma_0) \cdot H(z_0, h_0)\\
\cdots
\end{bmatrix}
$$

So, now the case there $x_0=x_1$ is a bit more complicated.  What we need is

$$
\begin{bmatrix}
I & 0 & 0 \\
\begin{bmatrix}
I & 0 \\
W_x & 0 \\
\end{bmatrix}
& 0 & \begin{bmatrix} 0 & 0 \\ 0 & c(\gamma_0) \end{bmatrix}\\
F & f & g 
\end{bmatrix}
$$
so we apply this matrix
$$
\begin{bmatrix}
I & 0 & 0 \\
\begin{bmatrix}
I & 0 \\
W_x & 0 \\
\end{bmatrix}
& 0 & \begin{bmatrix} 0 & 0 \\ 0 & c(\gamma_0) \end{bmatrix}\\
F & f & g 
\end{bmatrix}
\cdot
\begin{bmatrix}
z_0 = 
\begin{bmatrix}
x_0 \\
y_0 \\
\end{bmatrix} \\
z_0 \\
h_0 = 
\begin{bmatrix}
h_{x,0} \\
h_{y,0} \\
\end{bmatrix} \\
\end{bmatrix}
$$
to get
$$
\begin{bmatrix}
z_0 \\
\begin{bmatrix}
I & 0 \\
W_x & 0 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
x_0 \\
y_0 \\
\end{bmatrix} +
\begin{bmatrix} 
0 & 0 
\\ 0 & c(\gamma_0) 
\end{bmatrix} \cdot
\begin{bmatrix}
h_{x,0} \\
h_{y,0} \\
\end{bmatrix} 
\\
H
\end{bmatrix}
$$
which the same as
$$
\begin{bmatrix}
z_0 \\
\begin{bmatrix}
x_0 \\
W_x \cdot x_0 \\
\end{bmatrix} +
\begin{bmatrix}
0 \\
c(\gamma_0)\cdot h_{y,0} \\
\end{bmatrix} 
\\
H
\end{bmatrix}
$$
and we get the desired correction just on $y_0$
$$
\begin{bmatrix}
z_0 \\
\begin{bmatrix}
x_0 \\
W_x \cdot x_0 +c(\gamma_0)\cdot h_{y,0}\\
\end{bmatrix} 
\\
H
\end{bmatrix}
$$

and I don't think this case appears anywhere in my code!  Also, at this point, is this really a necessary optimization I mean $W_x$ is smaller that $W_z$ but seems way more complicated.

```python
    # At the first iteration we initialize the hidden state
    if state.i==0 and h_size is not None:
        state.h = torch.zeros(state.z.shape[0], h_size, device=state.z.device)

    # This conditional is kludgy
    if h_size is None:
        z_in = state.z
    else:
        z_in = torch.concat([state.z, state.h], dim=1)

    z_out = model(z_in)

    # This conditional is kludgy
    if h_size is not None:
        state.h = z_out[:, state.z.shape[1]:]  
        state.z = z_out[:, :state.z.shape[1]]
    else:
        state.z = z_out

    state.i += 1
 
    return state
```


# 6-17-2024


https://news.google.com/articles/CBMiSmh0dHBzOi8vd3d3Lm9tZ3VidW50dS5jby51ay8yMDI0LzA2L3RpbGluZy1zaGVsbC1nbm9tZS1leHRlbnNpb24taW4tdWJ1bnR10gFOaHR0cHM6Ly93d3cub21ndWJ1bnR1LmNvLnVrLzIwMjQvMDYvdGlsaW5nLXNoZWxsLWdub21lLWV4dGVuc2lvbi1pbi11YnVudHUvYW1w?hl=en-US&gl=US&ceid=US%3Aen

In the AI arena, progress towards true Artificial General Intelligence (AGI) has hit a wall.  <br>  <br>Current models can crunch data but can't adapt to new challenges. The ARC Prize 2024 competition invites participants to think ‘outside the bot’ to create AI that learns and thinks like humans–paving the way for revolutionary advancements.  <br>  <br>In this competition, you’ll develop AI systems that efficiently learn new skills and solve open-ended problems instead of depending exclusively on large language models (LLMs) trained with extensive datasets. Top submissions will show improvement toward human reasoning.  <br>  <br>Hosts: [Mike Knoop](https://notifications.google.com/g/p/ANiao5q-RytDEtjXia3Cr_aPjt-kkJOKFPfPNoef5aisFTthG00P5Ou6iHtxXtiJ7VZO_EvK6ID6GyMCBlPXOiftKg0bRyddsMj1L95p9lvcWlnPoyEeXQHk7g5f9iP3j-vnANNj40PK5xF6Xhw4F0MBfMoH6xHixjWcgWJoTPmWZswJd2oIwTdIpa5Cer5ARICtxk89) (Co-Founder, Zapier) & [Francois Chollet](https://notifications.google.com/g/p/ANiao5qzK1TSiWIPWEnNaRQMYdQO7ZhS029mldtDxV82NS5a9z_3CyjK9EkIUQHPhBzqfYd0yql8QeCiTS3in7ag0fKus2huvfbf2SQDKfb6cx7z3TyHUmmI-UASHcoVAkwU2cU6CTVlFSY5TnDhdOqn-F7WY9wTHNCA9Vz0Qlnmj4M1OsN9nB2LbTLV8HJcQZ1wpIGdLDvaLsQ) (Keras, creator of ARC-AGI) at [ARCprize](https://notifications.google.com/g/p/ANiao5qGF6a_iyuULon2V9wMivAc_QSbjn6R_IC2liBvItdRCXCG1LuOVlQxXJZMDIecWBOzvVyDgkQiIz7SB91Kzm5HsbocqFyO7yI_P3fJtbpppwmOm3ChUShuE3X7Exz6_aKDGKIJx5V60UOaxAIclLA_HeK_kZkU_s9ILQ_6O8JyPtSY-vy9Qpj6HPKcHFe1s09rlA)                                                                                                                                                                                                                                                          
 [Learn More](https://notifications.google.com/g/p/ANiao5rLDeGiIX5wJj3Ki0IoTR_Xf0rM9E0kXPpAczb8ZCIz3tiT3m1pLfkTlH8v4JmtG_ygsjIH-rSr3gOuZl1335w19cfHNk_RUAExDtyXavletZz9D9KeJYWdYODL-ERpw3to5CISducYjyr_K7qFYOTgwiZQeLN1YvseCXUxvUQCAoC_QwARsAVMtT97iAQnfu8VJmXqbHjhQqEpLDcwUTahkxuVC1dgDJ3vayqD6JZ1EyOV3Jasctx4peb9_y2I9I6CTbIrRyrvqMvsVc-Z1sIN-n90pOpaMXhF2AnxNkP9XDvhVPXAhisrC4Mc9j0JhQ "Learn More")                                                                                                                                
LS solution 0.0839

# 6-19-2024

This is what seems to work 

$$
z_{k+1} = W_{LS} \cdot z_0 + h(z_k) \cdot C
$$
where $h$ is a *half-square*.  I.e., the *half-square* works great on itself, and often beats the MLP, but with the above initialization it works even bettern.
One can also do

$$
z_{k+1} = W_{LS} \cdot z_0 + h(z_k) \odot c
$$

where $c$ is some row vector so $c$ has fewer unknowns than $C$.

I think the equivalent to our previous notation is

$$
\begin{bmatrix}
I & 0 & 0 \\
W_{LS} & 0 & C \\
0 & h & 0
\end{bmatrix}
$$
I think the lower-right corner was the problem, but I am not sure why?
The results are in

[inn_survey_KEEP_FOR_SIMON_6-19 Workspace – Weights & Biases (wandb.ai)](wandb.ai))

and the code that generated then is in

[Release csu_simon_results · rcpaffenroth/inn_survey (github.com)](github.com))

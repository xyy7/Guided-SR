def set_config(args):
    if args.sched == "multistep":
        args.decay_epochs = [int(x) for x in args.decay_epochs.split("_")]
    else:
        args.decay_epochs = int(args.decay_epochs)

    if args.exp_name == "":
        args.exp_name = args.model_name
    args.exp_name += (
        f"_S_{args.scale}_Loss_{args.loss}_LR_{args.lr}_Bs_{args.batch_size}_Ps_{args.patch_size}_Seed_{args.seed}"
    )

    # if args.rgb_norm:
    #     args.exp_name += "_Norm"
    # if args.mix_up:
    #     args.exp_name += f"_MX_{args.mix_alpha}"

    # if args.pre_trained:
    #     args.exp_name += "_PT"

    # if args.extra_data:
    #     args.exp_name += "_ED"

    # if args.no_res:
    #     args.exp_name += "_NR"

    # if args.with_noisy:
    #     args.exp_name += "_Noisy"

    # if args.random_down:
    #     args.exp_name += "_RD"

    # if args.mat_resize:
    #     args.exp_name += "_MR"

    if args.local_rank == 0:
        print(args)

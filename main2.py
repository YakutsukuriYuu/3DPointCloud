import controller.Tdeepla as deepla
import controller.Tpointmlp as pointmlp
import controller.Tpointnet as pointnet
import controller.Tdgcnn as dgcnn
import controller.Tunipre3d as unipre3d
import argparse
def main():
    parser = argparse.ArgumentParser(description="Select Model")

    parser.add_argument(
        "--mode",
        type=str,
        default="eval",
        choices=["train", "eval"],
        help="train() or eval()"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="none",
        choices=["deepla", "pointmlp", "pointnet", "dgcnn", "none","unipre3d"],
        help="Choose model"
    )

    args = parser.parse_args()  
    if args.mode == "train":
        if args.model=="deepla":
            print("deepla")
            deepla.train()
        elif args.model=="pointmlp":
            print("pointmlp")
            pointmlp.train()
        elif args.model=="pointnet":
            print("pointnet")
            pointnet.train()
        elif args.model=="dgcnn":
            print("dgcnn")
            dgcnn.train()
        elif args.model=="unipre3d":
            print("unipre3d")
            unipre3d.train()

    else:
        if args.model=="deepla":
            print("deepla")
            deepla.eval()
        elif args.model=="pointmlp":
            print("pointmlp")
            pointmlp.eval()
        elif args.model=="pointnet":
            print("pointnet")
            pointnet.eval()
        elif args.model=="dgcnn":
            print("dgcnn")
            dgcnn.eval()
        elif args.model=="unipre3d":
            print("unipre3d")
            unipre3d.eval()
    #deepla.eval()
    #deepla.train()

    #pointmlp.train()
    #pointmlp.eval()

    #pointnet.train()
    #pointnet.eval()
    #dgcnn.train()

if __name__=="__main__":
    main()

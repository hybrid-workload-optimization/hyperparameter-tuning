import rest
import controller
import argparse


# Parse CLI arguments
parser = argparse.ArgumentParser(description='Katib handling rest api server')
parser.add_argument('--host', type=str, default='0.0.0.0',
                    help="the hostname to listen on. Set this to '0.0.0.0'")
parser.add_argument('--port', type=int, default=6060,
                    help="the port of the webserver. Defaults to 6060")
parser.add_argument('--config', type=str, default='./config',
                    help='set config file of kubernetes cluster')
parser.add_argument('--debug', action='store_true',
                    help='the debug mode of rest api server. Defaults to False (True|False)')
opt = parser.parse_args()


if __name__ == "__main__":
    controller.api.init(opt.config)
    rest.register(controller.experiments)
    rest.register(controller.suggestion)
    rest.register(controller.base)
    rest.start(ip=opt.host, port=opt.port, debug=opt.debug)

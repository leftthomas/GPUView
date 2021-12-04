import argparse
import json
from datetime import datetime
import gevent.monkey
gevent.monkey.patch_all()
from bottle import Bottle, template

from gpuview import core

app = Bottle()


def arg_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='action', help='Action')

    run_parser = subparsers.add_parser('run', help='Run GPUView server')
    run_parser.add_argument('--host', required=True, type=str, help='IP address of host (e.g. 127.0.0.1)')
    run_parser.add_argument('--port', type=int, default=9988, help='Port number of host (default: 9988)')

    add_parser = subparsers.add_parser('add', help='Register a GPU host')
    add_parser.add_argument('--url', required=True, type=str, help='URL of GPU host (IP:Port, e.g. X.X.X.X:9988)')
    add_parser.add_argument('--name', default=None, type=str, help='An optional readable name for the GPU host')

    rem_parser = subparsers.add_parser('remove', help='Remove a GPU host')
    rem_parser.add_argument('--url', required=True, type=str, help='URL of the GPU host to remove')

    subparsers.add_parser('hosts', help='Print all GPU hosts')

    return parser


@app.route('/')
def index():
    gpu_stats = core.all_gpu_stats()
    now = datetime.now().strftime('Updated at %Y-%m-%d %H-%M-%S')
    return template('index', gpu_stats=gpu_stats, update_time=now)


@app.route('/gpustat', methods=['GET'])
def report_gpu_stat():
    """
    Returns the gpu stat of this host.
    """
    resp = core.local_gpu_stat()
    return json.dumps(resp)


def main():
    parser = arg_parser()
    args = parser.parse_args()

    if 'run' == args.action:
        app.run(server='gevent', host=args.host, port=args.port)
    elif 'add' == args.action:
        core.add_host(args.url, args.name)
    elif 'remove' == args.action:
        core.remove_host(args.url)
    elif 'hosts' == args.action:
        core.print_hosts()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

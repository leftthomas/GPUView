import argparse
import json
import os
from datetime import datetime

from bottle import Bottle, TEMPLATE_PATH, template, response

from . import core

app = Bottle()
abs_path = os.path.dirname(os.path.realpath(__file__))
abs_views_path = os.path.join(abs_path, 'views')
TEMPLATE_PATH.insert(0, abs_views_path)


def arg_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Action')

    run_parser = subparsers.add_parser('run', help='Run GPUView server')
    run_parser.add_argument('--host', required=True, type=str, help='IP address of host (eg. 0.0.0.0)')
    run_parser.add_argument('--port', type=int, default=9988, help='Port number of host (default: 9988)')

    add_parser = subparsers.add_parser('add', help='Register a new GPU host')
    add_parser.add_argument('--url', required=True, type=str, help='URL of GPU host (IP:Port, eg. X.X.X.X:9988')
    add_parser.add_argument('--name', default=None, type=str, help='An optional readable name for the GPU host')

    rem_parser = subparsers.add_parser('remove', help='Remove a GPU host')
    rem_parser.add_argument('--url', required=True, type=str, help='URL of the GPU host to remove')

    subparsers.add_parser('hosts', help='Print all GPU hosts')

    return parser


@app.route('/')
def index():
    gpustats = core.all_gpustats()
    now = datetime.now().strftime('Updated at %Y-%m-%d %H-%M-%S')
    return template('index', gpustats=gpustats, update_time=now)


@app.route('/gpustat', methods=['GET'])
def report_gpustat():
    """
    Returns the gpustat of this host.
        See `exclude-self` option of `gpuview run`.
    """

    def _date_handler(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        else:
            raise TypeError(type(obj))

    response.content_type = 'application/json'
    if EXCLUDE_SELF:
        resp = {'error': 'Excluded self!'}
    else:
        resp = core.my_gpustat()
    return json.dumps(resp, default=_date_handler)


def main():
    parser = arg_parser()
    args = parser.parse_args()

    if 'run' == args.action:
        core.safe_zone(args.safe_zone)
        global EXCLUDE_SELF
        EXCLUDE_SELF = args.exclude_self
        app.run(host=args.host, port=args.port, debug=args.debug)
    elif 'service' == args.action:
        core.install_service(host=args.host,
                             port=args.port,
                             safe_zone=args.safe_zone,
                             exclude_self=args.exclude_self)
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

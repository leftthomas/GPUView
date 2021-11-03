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

EXCLUDE_SELF = False  # Do not report to `/gpustat` calls.


def arg_parser():
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest='action', help='Action')

    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--host', default='0.0.0.0',
                             help='IP address of host (default: 0.0.0.0)')
    base_parser.add_argument('--port', default=9988,
                             help='Port number of host (default: 9988)')
    base_parser.add_argument('--safe-zone', action='store_true',
                             help='Report all details including usernames')
    base_parser.add_argument('--exclude-self', action='store_true',
                             help='Don\'t report to others but self-dashboard')
    run_parser = subparsers.add_parser('run', parents=[base_parser],
                                       help='Run gpuview server')
    run_parser.add_argument('-d', '--debug', action='store_true',
                            help='Run server in debug mode')
    add_parser = subparsers.add_parser('add', help='Register a new GPU host')
    add_parser.add_argument('--url', required=True,
                            help='URL of GPU host (IP:Port, eg. X.X.X.X:9988')
    add_parser.add_argument('--name', default=None,
                            help='An optional readable name for the GPU host')
    rem_parser = subparsers.add_parser('remove', help='Remove a GPU host')
    rem_parser.add_argument('--url', required=True,
                            help='Url of the GPU node to remove')
    subparsers.add_parser('hosts', help='Print all GPU hosts')
    subparsers.add_parser('service', parents=[base_parser],
                          help='Install gpuview as a service')

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

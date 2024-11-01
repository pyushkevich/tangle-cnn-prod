from .wildcat_main import parser

args = parser.parse_args()
args.func(args)
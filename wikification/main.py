import sys
sys.path.append('/home/user/AMRBART-v3/wikification/BLINK')

import warnings
warnings.simplefilter('ignore')

from amrlib.utils.logging import setup_logging, silence_penman, WARN
from amrlib.graph_processing.wiki_adder_blink import WikiAdderBlink


if __name__ == '__main__':
    setup_logging('logs/blink_wikify.log', level=WARN)
    silence_penman()

    model_dir  = 'BLINK/models'
    infpath    = 'data/input.amr'
    outfpath   = 'data/input.amr.wiki'

    # Load the BLINK models
    wa = WikiAdderBlink(model_dir)
    wa.wikify_file(infpath, outfpath)

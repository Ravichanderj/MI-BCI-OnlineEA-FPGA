// C-simulation test bench: integer decoder vs the int8 golden vector exported by deploy.py
#include <cstdio>
#include <cmath>
#include "hanet_cnn.h"
#include "export_S1/golden.h"
#include "export_S1/hanet_consts.h"

int main() {
    int32_t acc[CNN_NCLS];
    hanet_cnn(GOLDEN_IN, acc);
    int am_hw = 0, am_ref = 0;
    double maxerr = 0, maxref = 0;
    for (int n = 0; n < CNN_NCLS; n++) {
        double hw = acc[n] * (double)LOGIT_SCALE[n];
        printf("class %d: HLS %+.5f   python-int8 %+.5f\n", n, hw, GOLDEN_LOGITS[n]);
        maxerr = fmax(maxerr, fabs(hw - GOLDEN_LOGITS[n]));
        maxref = fmax(maxref, fabs(GOLDEN_LOGITS[n]));
        if (hw > acc[am_hw] * (double)LOGIT_SCALE[am_hw]) am_hw = n;
        if (GOLDEN_LOGITS[n] > GOLDEN_LOGITS[am_ref]) am_ref = n;
    }
    double rel = maxerr / maxref;
    printf("max |err| %.4f (%.2f%% of max |logit|), argmax HLS %d / python %d\n", maxerr, 100 * rel, am_hw, am_ref);
    bool ok = (am_hw == am_ref) && rel < 0.05;
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

#import "QWMetalCounterSupport.h"

NSData *QWResolveMetalCounterSamples(id<MTLCounterSampleBuffer> sampleBuffer,
                                     NSRange sampleRange) {
    return [sampleBuffer resolveCounterRange:sampleRange];
}

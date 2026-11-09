#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSData * _Nullable QWResolveMetalCounterSamples(
    id<MTLCounterSampleBuffer> sampleBuffer,
    NSRange sampleRange);

NS_ASSUME_NONNULL_END

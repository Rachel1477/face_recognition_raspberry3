from speaker_verifier_onnx import SpeakerVerifier

verifier = SpeakerVerifier(threshold=0.6)
similarity, is_same = verifier.verify('speaker1_1.mp3', 'speaker1_2.mp3')
print(f"相似度: {similarity:.4f}, 结果: {'同一人' if is_same else '不同人'}")
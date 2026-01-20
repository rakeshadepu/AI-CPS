# Use the busybox base image as required 
FROM busybox

# Create the required directory structure [cite: 19]
RUN mkdir -p /tmp/learningBase/train/ /tmp/learningBase/validation/

# Copy your data to the specific paths [cite: 19]
COPY data/training_data.csv /tmp/learningBase/train/
COPY data/test_data.csv /tmp/learningBase/validation/
COPY README.md /tmp/learningBase/

# Standard command to keep the container valid
CMD ["sh"]
# Use the busybox base image as required 
FROM busybox

# Create the required directory structure
RUN mkdir -p /tmp/activationBase/

# Copy your data to the specific paths
COPY activation_data.csv /tmp/activationBase/
COPY README.md /tmp/activationBase/

# Use the busybox base image
CMD ["sh"]


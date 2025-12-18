ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG SSH_PORT=22345

RUN rm -rf /etc/apt/sources.list.d/cuda-* && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        sudo \
        openssh-server \
        gdb \
        && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# Create or modify the user
RUN if getent group $USER_GID > /dev/null; then \
        EXISTING_GROUP=$(getent group $USER_GID | cut -d: -f1); \
        if [ "$EXISTING_GROUP" != "$USERNAME" ]; then \
            groupmod -n $USERNAME $EXISTING_GROUP; \
        fi; \
    else \
        groupadd --gid $USER_GID $USERNAME; \
    fi && \
    if id -u $USER_UID > /dev/null 2>&1; then \
        EXISTING_USER=$(getent passwd $USER_UID | cut -d: -f1); \
        if [ "$EXISTING_USER" != "$USERNAME" ]; then \
            usermod -l $USERNAME -d /home/$USERNAME -m $EXISTING_USER; \
        fi; \
    else \
        useradd --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi && \
    usermod -g $USERNAME $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN python3 -m pip install --no-cache-dir \
    pytest

RUN sed -i "s/^Port 22/Port ${SSH_PORT}/" /etc/ssh/sshd_config && \
    mkdir -p /home/$USERNAME/.ssh && \
    ssh-keygen -t rsa -f /home/$USERNAME/.ssh/id_rsa -N "" -q && \
    cat /home/$USERNAME/.ssh/id_rsa.pub >> /home/$USERNAME/.ssh/authorized_keys && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.ssh

# Add a check to .bashrc to fix "sudo: unable to resolve host" warning at runtime
RUN echo 'grep -q "$(hostname)" /etc/hosts || echo "127.0.0.1 $(hostname)" | sudo tee -a /etc/hosts > /dev/null 2>&1' >> /home/$USERNAME/.bashrc

USER $USERNAME

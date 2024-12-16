#FROM debian:bullseye-slim
FROM gcc:12.3.0
# make the "en_US.UTF-8" locale so postgres will be utf-8 enabled by default
ENV LANG en_US.utf8
ENV PG_MAJOR 13
ENV PG_VERSION 13.4
ENV PGDATA /u02/pgdata
ENV PGDATABASE "" \
    PGUSERNAME "" \
    PGPASSWORD ""

ARG UID=999
ARG GID=999

ENV PG_INSTALL_DIR /u01/app/postgres/product/${PG_VERSION}

RUN set -ex \
    \
    && apt-get update && apt-get install -y \
    gettext \
    sudo \
    ca-certificates \
    build-essential \
    curl \
    procps \
    sysstat \
    libldap2-dev \
    python3.11-dev \
    libreadline-dev \
    libssl-dev \
    bison \
    flex \
    libghc-zlib-dev \
    libcrypto++-dev \
    libxml2-dev \
    libxslt1-dev \
    bzip2 \
    make \
    gcc \
    unzip \
    python3.11 \
    locales \
    wget \
    cmake \
    python3.11-venv \
    \
  #  && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 en_US.UTF-8

RUN apt-get update && \
	apt-get install -y --no-install-recommends wget git golang-go  swig vim\
    libboost-filesystem-dev libboost-test-dev libboost-serialization-dev libboost-regex-dev libboost-serialization-dev libboost-regex-dev libboost-thread-dev libboost-system-dev

# COPY ./boost_1_81_0.tar.gz ./
# RUN tar -xzvf boost_1_81_0.tar.gz && \
#     cd boost_1_81_0 && \
#     ./bootstrap.sh && \
#     ./b2 install && \
#     ldconfig && \
#     cd .. && rm -rf boost_1_81_0
RUN wget "https://boostorg.jfrog.io/artifactory/main/release/1.81.0/source/boost_1_81_0.tar.gz" --no-check-certificate -q -O - \
        | tar -xz && \
        cd boost_1_81_0 && \
        ./bootstrap.sh && \
        ./b2 install && \
        ldconfig && \
        cd .. && rm -rf boost_1_81_0

RUN git config --global http.sslverify false

RUN mkdir /u01/ \
    \
    && groupadd -r postgres --gid=$GID \
    && useradd -m -r -s /bin/bash -g postgres --uid=$UID postgres \
    && chown postgres:postgres /u01/ \
    && mkdir -p "$PGDATA" \
    && chown -R postgres:postgres "$PGDATA" \
    && chmod 700 "$PGDATA" && echo "postgres:123" | chpasswd
    

# RUN useradd -m -s /bin/bash mr && \
#     echo "mr:mr" | chpasswd

RUN usermod -aG sudo postgres
RUN echo 'postgres ALL=(ALL) ALL' >> /etc/sudoers

COPY ./thirdparty/Postgres /home/postgres/src/
RUN echo "this line could be executed for a very long time" && chown -R postgres:postgres /home/postgres/src 
    
RUN cd /home/postgres/src \
    && su postgres -c "./configure \
    --with-blocksize=32 \
    --enable-integer-datetimes \
    --enable-thread-safety \
    --with-pgport=5432 \
    --prefix='/u01/app/postgres/product/13.4' \
    --with-ldap \
    --with-python \
    --with-openssl \
    --with-libxml \
    --with-libxslt \
    --enable-nls=yes" \
#    --enable-debug \
#    --enable-cassert \
#    CFLAGS='-ggdb -O0'" \
    && su postgres -c "make -j$(nproc) all" \
    && su postgres -c "make install" \
    && su postgres -c "make -C contrib install" \
    && apt-get install -y libxml2

ENV PGDATA ${PGDATA}/${PG_MAJOR}

COPY ./scripts/pg_scripts/*.sh /usr/local/bin/

ENV PATH="${PATH}:${PG_INSTALL_DIR}/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PG_INSTALL_DIR}/lib"

COPY . /tmp/vectordb

RUN chown -R postgres:postgres /tmp/vectordb

ENV PostgreSQL_ROOT ${PG_INSTALL_DIR}

RUN cd /tmp/vectordb && \
    mkdir build && \
	cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/vectordb -DLIBRARYONLY=ON -DSEEK_ENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release .. && \
	make -j$(nproc) && \
	make install

ENV PATH="${PATH}:/usr/local/lib"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

ENV LANG en_US.utf8
USER postgres

# WORKDIR /home/mr

# EXPOSE 5432
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["bash"]
#ENTRYPOINT ["sleep","infinity"]


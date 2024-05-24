FROM python
WORKDIR /app
COPY ./requirements.txt /app/
COPY ./main.py /app/
ENV SIMILAR_MATRIX='matrix.npy'
ENV RECOMMEND_ALGO='algo'
ENV RATINGS='ratings.csv'
ENV MOVIES='movies.csv'
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 3000
CMD ["python3", "main.py"]

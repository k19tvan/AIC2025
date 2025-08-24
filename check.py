from pymilvus import connections, Collection

# Kết nối tới Milvus
connections.connect(
    alias="default",
    host="milvus-standalone",
    port="19530"
)

# Mở collection
collection = Collection("Unite_Batch1_with_filepath_filter")
 z``
# Lấy số lượng entity
print(f"Số lượng entity trong collection '{collection.name}': {collection.num_entities}")

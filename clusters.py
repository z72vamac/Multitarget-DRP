class Cluster():
    
    def __init__(self):
        self.indices = []
        self.points = []
    
    def imprime(self):
        print(self.indices)
        print(self.points)
        

def pega_clusters(cluster1, cluster2):
    
    new_cluster = Cluster()
    new_cluster.indices = list( dict.fromkeys(cluster1.indices + cluster2.indices))
    
    return new_cluster
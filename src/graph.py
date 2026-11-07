class Graph:
    def __init__(self):
        self.adj = {}

    def number_of_nodes(self):
        return len(self.adj)

    def number_of_edges(self):
        return sum(len(neighbours) for neighbours in self.adj.values())
    
    # add_edge / degree unchanged
    def add_edge(self, station_start_id, station_end_id, count=1):
        if station_start_id not in self.adj:
            self.adj[station_start_id] = dict()
        if station_end_id not in self.adj:
            self.adj[station_end_id] = dict()
        self.adj[station_start_id][station_end_id] = self.adj[station_start_id].get(station_end_id, 0) + count

    def degree(self, station_id):
        if station_id not in self.adj:
            return 0
        return sum(self.adj[station_id].values())
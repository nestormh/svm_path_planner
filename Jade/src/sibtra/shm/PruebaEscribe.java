package sibtra.shm;

public class PruebaEscribe {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		ShmInterface.safeGet();
		
		int dist = 10;
		while(true) {
			ShmInterface.setAcoRightDist(dist);		
			System.out.println ( System.currentTimeMillis()+": Escribo Distancia " + dist);

			try{ Thread.sleep(500); } catch (Exception e) {}
			dist+=10;
		}

	}

}

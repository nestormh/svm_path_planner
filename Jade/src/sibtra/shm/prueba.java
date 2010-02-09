package sibtra.shm;

public class prueba extends Thread{


		public static void main(String[] args) {
			
			
		
			ShmInterface.safeGet();
		
			while(true) {
				int dist = ShmInterface.getAcoRightDist();		
				System.out.println ( System.currentTimeMillis()+": Distancia " + dist);

				try{ Thread.sleep(1000); } catch (Exception e) {}
			}
			
		}
		

}

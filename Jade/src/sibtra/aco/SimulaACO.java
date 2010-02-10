/**
 * 
 */
package sibtra.aco;

import sibtra.shm.ShmInterface;

/**
 * Mete datos en memoria compartida como si fuera ACO de C
 * @author alberto
 *
 */
public class SimulaACO {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		int derecha=10000;
		int izquierda=0;
		int orientacion=-10000;
		System.out.println("Comienza SimulaACO");
		while(true)
			try {
				Thread.sleep(500);
				ShmInterface.setAcoRightDist(derecha--);
				ShmInterface.setAcoLeftDist(izquierda++);
				ShmInterface.setAcoRoadOrientation(orientacion);
				orientacion+=2;
				ShmInterface.setAcoAlive(1);
				if((derecha%10)==0)
					System.out.println("Decha:"+derecha+" Izda:"+izquierda+" Orientacion:"+orientacion);
			} catch (Exception e) {
				// TODO: handle exception
			}		
	}

}

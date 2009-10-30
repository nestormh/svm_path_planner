/* marco para la gráfica del error */ 

package sibtra.ultrasonidos;


import java.awt.Button;
import java.awt.Color;
import java.awt.Frame;
import java.awt.GridLayout;
import java.awt.Label;
import java.awt.Panel;
import java.awt.TextField;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.util.Arrays;
import java.util.Hashtable;


/**
 * marco para representar la gráfica del error  
 * @author Evelio José González González
 * @version 15-09-99
 */
public class PlanoVisual extends Frame implements WindowListener {
	/* declaración del botón de cancelar */
	/**
	 * botón de cancelar 
	 */
	public TextField[][] campos;

	public double[][] radioCmAngulo;
	public double[][] radioPixelAngulo;
	public int numeroPuntos;

	/**
	 * menú para la posibilidad de ampliar la gráfica
	 */ 
	Button actualizar = new Button ("Actualizar");

	DibujoVisual graph;
	int inicio = 0;
	Ultrasonido[] ultras; 
	Carro carro;
	double coefA = 0.01732372;
	double coefB = -1.7942;

	public PlanoVisual(Ultrasonido[] ultras, Carro carro) {
		this.ultras=ultras;
		this.carro = carro;
		campos = new TextField[ultras.length][4];   
		for (int i=0; i<campos.length; i++) {
			for (int j=0; j<2; j++) {
				campos[i][j]= new TextField("255", 3);
			}
			campos[i][2] = new TextField(String.valueOf(256*Integer.parseInt(campos[i][0].getText()) + Integer.parseInt(campos[i][1].getText())), 5);
			campos[i][3] = new TextField(String.valueOf(coefA*Double.parseDouble(campos[i][2].getText()) + coefB), 5);
		}

		numeroPuntos = 5;
		radioCmAngulo = new double[ultras.length*numeroPuntos][2];
		radioPixelAngulo = new double[ultras.length*numeroPuntos][2];

		/* declaración de un color, que se pone como fondo */
		Color t = new Color(210,210,255);
		setBackground(t);
		addWindowListener(this);
		/* dibujo de la gráfica mediante la clase Dibujo, */
		/* al que se le pasa como argumento el vector de */
		/* error */
		graph=new DibujoVisual(ultras, carro);
		/* título del marco */
		setTitle("Gestor de ultrasonidos");
		/* añadir al marco el texto de area, el dibujo */
		/* de la gráfica y el botón de cancelar */
		/*  add("North",texto1); */
		add(graph);
		/*   add("East",cancelar);*/
		/* imprimir el error de la primera iteración */   



		/* declaración de un panel con los campos de texto */
		/* y los botones */
		Panel q = new Panel();
		q.setLayout(new GridLayout(2 + campos.length, 1 + campos[0].length));

		q.add(new Label("ID " ));
		q.add(new Label("Byte alto" ));
		q.add(new Label("Byte bajo" ));
		q.add(new Label("cm" ));


		for (int j=0; j<campos.length; j++) {
			q.add(new Label(String.valueOf (j+1)));
			for (int k=0; k<campos[0].length; k++) {
				if( k!= 2) {
					q.add(campos[j][k]);
				}
				if( k==3) {
					campos[j][k].setEditable(false);
				}
			}

		}
		actualizar.addActionListener(new Actualizar());  
		q.add(new Label(" "));

		q.add(actualizar);
		add("East", q); /* añadir el panel al marco */


	}


	/**
	 * método sin implementar
	 */
	public void windowIconified(WindowEvent evt) { }

	/**
	 * método sin implementar
	 */
	public void windowDeiconified(WindowEvent evt) { }

	/**
	 * método sin implementar
	 */
	public void windowOpened(WindowEvent evt) { }

	/**
	 * método sin implementar
	 */
	public void windowClosed(WindowEvent evt) { }

	/**
	 * cerrar el marco sin salir del programa
	 */
	public void windowClosing(WindowEvent evt) {
		dispose();
	}

	/**
	 * método sin implementar
	 */
	public void windowDeactivated(WindowEvent evt) { }

	/**
	 * método sin implementar
	 */
	public void windowActivated(WindowEvent evt) { }



	public void calculoDistancias() { }

	class Actualizar implements ActionListener {
		public void actionPerformed(ActionEvent e) {
			for(int i=0; i<campos.length; i++) {
				campos[i][2].setText(String.valueOf(256*Integer.parseInt(campos[i][0].getText()) + Integer.parseInt(campos[i][1].getText())));
				double radio = coefA*Double.parseDouble(campos[i][2].getText()) + coefB;
				campos[i][3].setText(String.valueOf(radio));
				ultras[i].fijaRadio(Math.min(20000,radio));
				ultras[i].volcarDatos();
				double L = ultras[i].dimePosicion().x - 300;
				double R1 = ultras[i].radio;
//				256*Double.parseDouble(campos[i][0].getText()) + Double.parseDouble(campos[i][1].getText());
//				System.out.println("L[" + i + "]=" + L);
//				System.out.println("R1[" + i + "]=" + R1);

				for (int q= 0; q < numeroPuntos; q++) {
					int indice = i*numeroPuntos + q;
					double Theta = Math.PI * (90 + 15 - (30* q /numeroPuntos))/180 + ultras[i].orientacion - carro.orientacion; 
					double radio0 = Math.sqrt(L*L + R1 * R1 - 2 * R1 * Math.abs(L) * Math.cos(Theta));  

					double angulo = Math.asin((R1/radio0)*Math.sin(Theta));
					if(L>0) {
						angulo = Math.asin((R1/radio0)*Math.sin(Theta));
					}
					if(L<0) {
						angulo =  Math.PI - Math.asin((R1/radio0)*Math.sin(Theta));
					}

//					radioCmAngulo[m][0]= 
					radioPixelAngulo[indice][0] = radio0;
					radioPixelAngulo[indice][1] = angulo;
					if (i < 8) {
//						System.out.println("radio[" + i + "]=" + radio0);
//						System.out.println("angulo[" + i + "]=" + angulo);
					}
				}
			}
			double[][] rPAorden = ordenar(radioPixelAngulo);
			graph.fijaContorno(rPAorden);
			graph.repaint();
		}
	}

	public double[][] ordenar (double[][] original) {
		double[][] ordenado = new double[8*numeroPuntos][2];
		Hashtable tabla = new Hashtable();
		double[] angulos = new double[ordenado.length];
		for (int j = 0; j < (8 * numeroPuntos); j++) {
			Object nulo = tabla.put(new Double(original[j][1]), new Double(original[j][0]));
			angulos[j] = original[j][1];
		}
		Arrays.sort(angulos);
		for(int i= 0; i < angulos.length; i++) {
			ordenado[i][0]= ((Double)(tabla.get(new Double(angulos[i])))).doubleValue();
			ordenado[i][1]= angulos[i];
//			System.out.println("AN = " + angulos[i] + "--" + ordenado[i][0]);
		}

		return ordenado;
	}


	public void actualizar() {
		for(int i=0; i<campos.length; i++) {
			campos[i][2].setText(String.valueOf(256*Integer.parseInt(campos[i][0].getText()) + Integer.parseInt(campos[i][1].getText())));
			double radio = coefA*Double.parseDouble(campos[i][2].getText()) + coefB;
			campos[i][3].setText(String.valueOf(radio));
			ultras[i].fijaRadio(Math.min(20000,radio));
			double L = ultras[i].dimePosicion().x -300;
			double R1 = 256*Double.parseDouble(campos[i][0].getText()) + Double.parseDouble(campos[i][1].getText());
			for (int q= 0; q < numeroPuntos; q++)
			{int indice = i*numeroPuntos + q;
			double Theta = Math.PI * (90 + 15 - (30* q /numeroPuntos))/180; 
			double radio0 = Math.sqrt(L*L + R1 * R1 - 2 * R1 * L * Math.cos(Theta));  
			double angulo = 0;
			if(L>0) {
				angulo = Math.asin((R1/radio0)*Math.sin(Theta));
			}
			if(L<0) {
				angulo =  Math.PI - Math.asin((R1/radio0)*Math.sin(Theta));
			}

//			radioCmAngulo[m][0]= 
			radioPixelAngulo[indice][0] = radio0;
			radioPixelAngulo[indice][1] = angulo;
			if (i < 8) {
//				System.out.println("radio[" + i + "]=" + radio0);
//				System.out.println("angulo[" + i + "]=" + angulo);
			}
			}
			double[][] rPAorden = ordenar(radioPixelAngulo);
			graph.fijaContorno(radioPixelAngulo);
			graph.repaint();
		}
	}

	public void actualizar(int i, int j, int k) {
		for(int m=0; m<ultras.length; m++) {
			int ident = ultras[m].dimeId();
			if (i==ident) {
				System.out.println("ID=" + ident + "; cuenta1 = " + j + " cuanta 2= " + k);  
				campos[m][0].setText(String.valueOf(j));
				campos[m][1].setText(String.valueOf(k));
				campos[m][2].setText(String.valueOf(256*Integer.parseInt(campos[m][0].getText()) + Integer.parseInt(campos[m][1].getText())));
				double radio = coefA*Double.parseDouble(campos[m][2].getText()) + coefB;
				campos[m][3].setText(String.valueOf(radio));
				ultras[m].fijaRadio(Math.min(20000,radio/2));
				double L = ultras[m].dimePosicion().x - 300;
				double R1 = 256*Double.parseDouble(campos[m][0].getText()) + Double.parseDouble(campos[m][1].getText());
				for (int q= 0; q < numeroPuntos; q++) {
					int indice = m*numeroPuntos + q;
					double Theta = Math.PI * (90 + 15 - (30* q /numeroPuntos))/180; 
					double radio0 = Math.sqrt(L*L + R1 * R1 - 2 * R1 * L * Math.cos(Theta));  
					double angulo = Math.asin((R1/radio0)*Math.sin(Theta));
//					radioCmAngulo[m][0]= 
					radioPixelAngulo[indice][0] = radio0;
					radioPixelAngulo[indice][1] = angulo;
					if (m < 8) {
//						System.out.println("radio[" + m + "]=" + radio0);
//						System.out.println("angulo[" + m + "]=" + angulo);

					}
				}


			}
			double[][] rPAorden = ordenar(radioPixelAngulo);
			graph.fijaContorno(radioPixelAngulo);
			graph.repaint();
		}

	}



}  


